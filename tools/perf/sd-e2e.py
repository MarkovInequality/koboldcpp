#!/usr/bin/env python3
# End-to-end image generation check through a real koboldcpp server, without downloaded weights: fixed-seed
# generations with synthetic models, recording pixel hashes and wall times.
#
#   sd-e2e.py record GOLDEN.json [options]   run and write the golden file
#   sd-e2e.py check  GOLDEN.json [options]   run and compare: hashes must match, times are reported against it
#   sd-e2e.py run [options]                  run and print only
#   sd-e2e.py compare DIR_A DIR_B            per-image PSNR and largest channel difference of two --save runs
#
# --build DIR    tree with koboldcpp.py and koboldcpp_cublas.so (default: this repo)
# --lib FILE     run a symlinked copy of --build with FILE as koboldcpp_cublas.so
# --save DIR     also write every output there
#
# qwen21: Qwen Image 2.1 with the synthetic models of test-sd-qwen21 --write-models (a tiny diffusion model, a tiny
# Qwen3-VL encoder with its vision tower in the same file, the published 2.1 VAE layout with deterministic values), flash
# attention, every model on the GPU: a text-to-image twice (the repeat must be bitwise equal, the output RGBA) and an edit
# of it through the vision encoder. Build the generator first: make test-sd-qwen21.

import argparse, base64, hashlib, json, math, os, shutil, signal, struct, subprocess, sys, tempfile, time, urllib.request, zlib

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
GENERATOR = os.path.join(REPO, "test-sd-qwen21")
BASE = {"seed": 42, "sampler_name": "euler", "negative_prompt": "", "width": 128, "height": 128, "steps": 4, "cfg_scale": 4.0}
Q21_T2I = dict(BASE, prompt="A red fox sitting in a snowy pine forest at dawn.")
Q21_EDIT = dict(BASE, prompt="Make the fox's fur bright blue.")

def servers(models):
    return {"qwen21": ["--sdmodel", f"{models}/dit.safetensors", "--sdvae", f"{models}/vae.safetensors",
                       "--sdllm", f"{models}/llm.safetensors", "--sdflashattention", "--sdclipdevice", "main"]}

class Server:
    def __init__(self, tree, flags, port, log):
        self.port = port
        self.logf = open(log, "w")
        cmd = [sys.executable, os.path.join(tree, "koboldcpp.py"), "--usecuda", "normal", "0", "--port", str(port),
               "--quiet", "--skiplauncher"] + flags
        self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT, cwd=tree, start_new_session=True)
        for _ in range(900):
            time.sleep(1)
            if self.proc.poll() is not None:
                raise RuntimeError(f"koboldcpp exited during load, see {log}")
            if "Please connect to custom endpoint" in open(log, errors="replace").read():
                return
        raise RuntimeError(f"koboldcpp did not come up, see {log}")

    def gen(self, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/sdapi/v1/txt2img", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
        wall = time.perf_counter() - t0
        data = base64.b64decode(r["images"][0]) if r.get("images") and r["images"][0] else b""
        return data, wall

    def stop(self):
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=60)
        except Exception:
            os.killpg(self.proc.pid, signal.SIGKILL)
        self.logf.close()

def png_pixels(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    pos, idat = 8, b""
    while pos < len(data):
        n, typ = struct.unpack(">I", data[pos:pos + 4])[0], data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        pos += 12 + n
        if typ == b"IHDR":
            w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", body)
        elif typ == b"IDAT":
            idat += body
        elif typ == b"IEND":
            break
    assert depth == 8 and interlace == 0 and ctype in (2, 6), (depth, interlace, ctype)
    ch = 3 if ctype == 2 else 4
    raw, stride = zlib.decompress(idat), w * ch
    out, prev, i = bytearray(), bytearray(stride), 0
    for _ in range(h):
        f, line = raw[i], bytearray(raw[i + 1:i + 1 + stride])
        i += 1 + stride
        for x in range(stride):
            a = line[x - ch] if x >= ch else 0
            if f == 1:
                line[x] = (line[x] + a) & 255
            elif f == 2:
                line[x] = (line[x] + prev[x]) & 255
            elif f == 3:
                line[x] = (line[x] + (a + prev[x]) // 2) & 255
            elif f == 4:
                b, c = prev[x], prev[x - ch] if x >= ch else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        out += line
        prev = line
    return w, h, ch, bytes(out)

def stddev(vals):
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

def record(res, key, data, wall, save, checks=()):
    px = png_pixels(data)
    entry = {"hash": hashlib.sha256(px[3] if px else data).hexdigest()[:16], "wall": round(wall, 2), "bytes": len(data)}
    if px:
        entry.update(w=px[0], h=px[1], ch=px[2])
    ok = bool(data)
    notes = []
    if px:
        rgb = [v for j, v in enumerate(px[3]) if j % px[2] < 3][::97]
        if stddev(rgb) < 8:
            ok, notes = False, notes + ["flat image"]
    for name, fn in checks:
        good = fn(px)
        ok &= good
        notes.append(f"{name} {'ok' if good else 'FAIL'}")
    entry["ok"] = ok
    res[key] = entry
    shape = f"{px[0]}x{px[1]}x{px[2]}" if px else f"{len(data)} bytes"
    print(f"  {key:10s} {entry['hash']} {shape:12s} {wall:7.2f}s {'; '.join(notes)}{'' if ok else '  FAIL'}", flush=True)
    if save:
        open(os.path.join(save, key + (".png" if px else ".bin")), "wb").write(data)
    return px

def run_qwen21(s, res, save):
    data, wall = s.gen(Q21_T2I)
    px = record(res, "t2i", data, wall, save, [("128x128 RGBA", lambda p: p and p[:3] == (128, 128, 4))])
    data2, wall = s.gen(Q21_T2I)
    record(res, "t2i-again", data2, wall, save, [("equal to t2i", lambda p: p and px and p[3] == px[3])])
    data, wall = s.gen(dict(Q21_EDIT, extra_images=[base64.b64encode(data2).decode()]))
    record(res, "edit", data, wall, save, [("differs from source", lambda p: p and px and p[3] != px[3])])

def compare(a, b):
    ok = True
    for name in sorted(os.listdir(a)):
        if not name.endswith(".png") or not os.path.exists(os.path.join(b, name)):
            continue
        pa, pb = (png_pixels(open(os.path.join(d, name), "rb").read()) for d in (a, b))
        if pa[:3] != pb[:3]:
            print(f"  {name:16s} shape {pa[:3]} vs {pb[:3]}  FAIL")
            ok = False
            continue
        diffs = [abs(x - y) for x, y in zip(pa[3], pb[3])]
        mse = sum(d * d for d in diffs) / len(diffs)
        psnr = float("inf") if mse == 0 else 10 * math.log10(255 * 255 / mse)
        print(f"  {name:16s} PSNR {psnr:6.2f} dB, largest difference {max(diffs):3d}, differing values {sum(1 for d in diffs if d) / len(diffs):6.2%}")
    return ok

def farm(build, lib, workdir):
    tree = os.path.join(workdir, "tree")
    os.makedirs(tree)
    for name in os.listdir(build):
        if name in ("koboldcpp.py", "koboldcpp_cublas.so"):
            continue
        os.symlink(os.path.join(build, name), os.path.join(tree, name))
    shutil.copy(os.path.join(build, "koboldcpp.py"), tree)  # it finds its library through realpath(__file__)
    os.symlink(os.path.realpath(lib), os.path.join(tree, "koboldcpp_cublas.so"))
    return tree

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "check", "run", "compare"])
    ap.add_argument("golden", nargs="?")
    ap.add_argument("other", nargs="?")
    ap.add_argument("--build", default=REPO)
    ap.add_argument("--lib")
    ap.add_argument("--port", type=int, default=5099)
    ap.add_argument("--save")
    args = ap.parse_args()
    if args.mode == "compare":
        ok = compare(args.golden, args.other)
        return 0 if ok else 1
    if args.mode != "run" and not args.golden:
        ap.error("record and check need a golden file")
    if not os.path.exists(GENERATOR):
        sys.exit(f"{GENERATOR} not found; make test-sd-qwen21")
    if args.save:
        os.makedirs(args.save, exist_ok=True)

    workdir = tempfile.mkdtemp(prefix="sd-e2e-")
    models = os.path.join(workdir, "models")
    subprocess.run([GENERATOR, "--write-models", models], cwd=REPO, check=True)
    tree = farm(os.path.realpath(args.build), args.lib, workdir) if args.lib else os.path.realpath(args.build)
    res, ok = {}, True
    for name, flags in servers(models).items():
        print(f"{name}:", flush=True)
        save = os.path.join(args.save, name) if args.save else None
        if save:
            os.makedirs(save, exist_ok=True)
        s = Server(tree, flags, args.port, os.path.join(workdir, f"{name}.log"))
        try:
            res[name] = {}
            run_qwen21(s, res[name], save)
        finally:
            s.stop()
        ok &= all(e["ok"] for e in res[name].values())
    shutil.rmtree(models)
    print(f"logs in {workdir}")

    if args.mode == "record":
        json.dump(res, open(args.golden, "w"), indent=1)
        print(f"recorded {args.golden}")
    elif args.mode == "check":
        gold = json.load(open(args.golden))
        for c, d in res.items():
            for k, r in d.items():
                g = gold.get(c, {}).get(k)
                if not g:
                    continue
                same = g["hash"] == r["hash"]
                ok &= same
                print(f"  {c:7s} {k:10s} {'same' if same else 'DIFF'} hash, wall {r['wall']:7.2f}s vs {g['wall']:7.2f}s "
                      f"({g['wall'] / r['wall']:5.3f}x){'' if same else '  FAIL'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
