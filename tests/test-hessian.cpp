// Checks on hessian-collect's output files, for tools/hessian/tests.sh. Grams are read whole: small models only.
//
//   test-hessian check FILE [--weights N]     format: every Gram and alias loads, G symmetric, finite and with a
//                                             non-negative diagonal, each weight's in_sum2 = diag G bitwise, counts
//                                             equal within a group; N: expected number of weights
//   test-hessian compare A B [--tol T] [--per-count] [--frob]
//                                             max over entries of |G_A - G_B| / sqrt(G_B,ii G_B,jj) (Grams divided by
//                                             their counts with --per-count), or with --frob the largest relative
//                                             Frobenius difference of a Gram; fails above T
//   test-hessian sum A B C [--tol T]          G_A + G_B = G_C within T (same measure), counts add up
//   test-hessian diag FILE IMATRIX [--tol T]  in_sum2 agrees with an imatrix's, per weight, within T in relative L2
//                                             (the largest elementwise difference is reported), with equal counts
//   test-hessian imatrix-load FILE            common_imatrix_load on FILE: entries and peak RSS
//   test-hessian count FILE WEIGHT [N]        prints the rows summed into WEIGHT's Gram; fails unless it is N

#include "llama-hessian.h"
#include "common/imatrix-loader.h"

#include <cinttypes>
#include <sys/resource.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static double opt(int argc, char ** argv, const char * name, double def) {
    for (int i = 0; i + 1 < argc; ++i) {
        if (!strcmp(argv[i], name)) {
            return atof(argv[i + 1]);
        }
    }
    return def;
}

static bool flag(int argc, char ** argv, const char * name) {
    for (int i = 0; i < argc; ++i) {
        if (!strcmp(argv[i], name)) {
            return true;
        }
    }
    return false;
}

static bool load(const char * path, llama_hessian & h) {
    if (!h.open(path)) {
        fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    return true;
}

static std::vector<std::string> owners_of(const llama_hessian & h) {
    std::vector<std::string> out;
    for (const auto & w : h.weights()) {
        if (h.owner(w) == w) {
            out.push_back(w);
        }
    }
    return out;
}

static int cmd_check(int argc, char ** argv) {
    llama_hessian h;
    if (!load(argv[2], h)) {
        return 1;
    }
    const auto ws = h.weights();
    const auto os = owners_of(h);
    int bad = 0;
    std::vector<float> g, s2;
    for (const auto & o : os) {
        const int64_t n = h.n(o);
        g.resize((size_t) n*n);
        if (!h.read(o, g.data())) {
            fprintf(stderr, "%s: cannot read its Gram\n", o.c_str());
            ++bad;
            continue;
        }
        bool sym = true, fin = true, pos = true;
        for (int64_t i = 0; i < n; ++i) {
            pos &= g[i*n + i] >= 0.0f;
            for (int64_t j = 0; j < n; ++j) {
                fin &= std::isfinite(g[i*n + j]);
                sym &= memcmp(&g[i*n + j], &g[j*n + i], sizeof(float)) == 0;
            }
        }
        if (!sym || !fin || !pos) {
            fprintf(stderr, "%s: symmetric %d, finite %d, non-negative diagonal %d\n", o.c_str(), sym, fin, pos);
            ++bad;
        }
        for (const auto & w : ws) {
            if (h.owner(w) != o) {
                continue;
            }
            s2.resize(n);
            if (!h.read_sum2(w, s2.data())) {
                fprintf(stderr, "%s: no in_sum2\n", w.c_str());
                ++bad;
                continue;
            }
            for (int64_t i = 0; i < n; ++i) {
                if (memcmp(&s2[i], &g[i*n + i], sizeof(float)) != 0) {
                    fprintf(stderr, "%s: in_sum2[%" PRId64 "] = %g != diag %g\n", w.c_str(), i, s2[i], g[i*n + i]);
                    ++bad;
                    break;
                }
            }
            if (h.count(w) != h.count(o)) {
                fprintf(stderr, "%s: count %g != its owner's %g\n", w.c_str(), h.count(w), h.count(o));
                ++bad;
            }
        }
    }
    const int want = (int) opt(argc, argv, "--weights", -1);
    printf("%s: %zu weights, %zu Grams, complete %d%s\n", argv[2], ws.size(), os.size(), h.complete, bad ? ", FAILED" : "");
    if (want >= 0 && (int) ws.size() != want) {
        fprintf(stderr, "expected %d weights\n", want);
        ++bad;
    }
    return bad ? 1 : 0;
}

// max |a - b| / sqrt(b_ii b_jj) over one Gram
static double gram_err(const std::vector<float> & a, double ca, const std::vector<float> & b, double cb, int64_t n) {
    double worst = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            const double d  = std::fabs(a[i*n + j]/ca - b[i*n + j]/cb);
            const double sc = std::sqrt((double) b[i*n + i]/cb * (double) b[j*n + j]/cb);
            if (sc > 0) {
                worst = std::max(worst, d/sc);
            }
        }
    }
    return worst;
}

static int cmd_compare(int argc, char ** argv) {
    llama_hessian A, B;
    if (!load(argv[2], A) || !load(argv[3], B)) {
        return 1;
    }
    const double tol = opt(argc, argv, "--tol", 1e-5);
    const bool per_count = flag(argc, argv, "--per-count");
    const bool frob = flag(argc, argv, "--frob");
    double worst = 0.0, worst_f = 0.0;
    std::string worst_w, worst_fw;
    int n_cmp = 0;
    std::vector<float> a, b;
    for (const auto & o : owners_of(B)) {
        if (!A.has(o)) {
            fprintf(stderr, "%s: missing from %s\n", o.c_str(), argv[2]);
            return 1;
        }
        const int64_t n = B.n(o);
        a.resize((size_t) n*n);
        b.resize((size_t) n*n);
        A.read(o, a.data());
        B.read(o, b.data());
        const double ca = per_count ? A.count(o) : 1.0, cb = per_count ? B.count(o) : 1.0;
        if (!per_count && A.count(o) != B.count(o)) {
            fprintf(stderr, "%s: counts differ (%g vs %g)\n", o.c_str(), A.count(o), B.count(o));
            return 1;
        }
        const double e = gram_err(a, ca, b, cb, n);
        if (e > worst) {
            worst = e;
            worst_w = o;
        }
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double d = a[i]/ca - b[i]/cb;
            num += d*d;
            den += (double) b[i]/cb * b[i]/cb;
        }
        const double f = den > 0 ? std::sqrt(num/den) : 0.0;
        if (f > worst_f) {
            worst_f = f;
            worst_fw = o;
        }
        ++n_cmp;
    }
    const double got = frob ? worst_f : worst;
    printf("compare %s %s: %d Grams, max |dG_ij|/sqrt(G_ii G_jj) = %.3g (%s), max relative Frobenius %.3g (%s), %s tolerance %.3g: %s\n",
           argv[2], argv[3], n_cmp, worst, worst_w.c_str(), worst_f, worst_fw.c_str(), frob ? "Frobenius" : "entry", tol,
           got <= tol ? "ok" : "FAILED");
    return got <= tol ? 0 : 1;
}

static int cmd_sum(int argc, char ** argv) {
    llama_hessian A, B, C;
    if (!load(argv[2], A) || !load(argv[3], B) || !load(argv[4], C)) {
        return 1;
    }
    const double tol = opt(argc, argv, "--tol", 1e-6);
    double worst = 0.0;
    std::string worst_w;
    std::vector<float> a, b, c;
    for (const auto & o : owners_of(C)) {
        const int64_t n = C.n(o);
        a.resize((size_t) n*n);
        b.resize((size_t) n*n);
        c.resize((size_t) n*n);
        A.read(o, a.data());
        B.read(o, b.data());
        C.read(o, c.data());
        if (A.count(o) + B.count(o) != C.count(o)) {
            fprintf(stderr, "%s: counts %g + %g != %g\n", o.c_str(), A.count(o), B.count(o), C.count(o));
            return 1;
        }
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] += b[i];
        }
        const double e = gram_err(a, 1.0, c, 1.0, n);
        if (e > worst) {
            worst = e;
            worst_w = o;
        }
    }
    printf("sum: max |G_A + G_B - G_C|/sqrt(G_ii G_jj) = %.3g (%s), tolerance %.3g: %s\n", worst, worst_w.c_str(), tol, worst <= tol ? "ok" : "FAILED");
    return worst <= tol ? 0 : 1;
}

static int cmd_diag(int argc, char ** argv) {
    llama_hessian h;
    common_imatrix im;
    if (!load(argv[2], h) || !common_imatrix_load(argv[3], im)) {
        return 1;
    }
    const double tol = opt(argc, argv, "--tol", 5e-3);
    double worst = 0.0, worst_l2 = 0.0;
    std::string worst_w, worst_l2w;
    int n_cmp = 0, bad = 0;
    std::vector<float> s2;
    for (const auto & [name, e] : im.entries) {
        if (!h.has(name)) {
            continue;
        }
        const int64_t n = h.n(name);
        s2.resize(n);
        h.read_sum2(name, s2.data());
        const double c = h.count(name);
        if ((int64_t) e.sums.size() != n || e.counts.size() != 1 || (double) e.counts[0] != c) {
            fprintf(stderr, "%s: size %zu vs %" PRId64 ", counts %" PRId64 " vs %g\n", name.c_str(), e.sums.size(), n,
                    e.counts.empty() ? (int64_t) -1 : e.counts[0], c);
            ++bad;
            continue;
        }
        double mean = 0.0, num = 0.0, den = 0.0;
        for (int64_t j = 0; j < n; ++j) {
            mean += e.sums[j];
            num += ((double) s2[j] - e.sums[j])*((double) s2[j] - e.sums[j]);
            den += (double) e.sums[j]*e.sums[j];
        }
        mean /= n;
        const double l2 = den > 0 ? std::sqrt(num/den) : 0.0;
        if (l2 > worst_l2) {
            worst_l2 = l2;
            worst_l2w = name;
        }
        for (int64_t j = 0; j < n; ++j) {
            // relative, with a floor at 1e-6 of the row's mean for near-dead channels
            const double r = std::fabs(s2[j] - e.sums[j]) / std::max((double) std::fabs(e.sums[j]), 1e-6*mean);
            if (r > worst) {
                worst = r;
                worst_w = name;
            }
        }
        ++n_cmp;
    }
    printf("diag: %d weights compared, max relative L2 difference %.3g (%s), largest elementwise %.3g (%s), tolerance %.3g: %s\n",
           n_cmp, worst_l2, worst_l2w.c_str(), worst, worst_w.c_str(), tol, (worst_l2 <= tol && !bad && n_cmp > 0) ? "ok" : "FAILED");
    return (worst_l2 <= tol && !bad && n_cmp > 0) ? 0 : 1;
}

static int cmd_imatrix_load(char ** argv) {
    common_imatrix im;
    if (!common_imatrix_load(argv[2], im)) {
        return 1;
    }
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    printf("imatrix-load %s: %zu entries, peak RSS %.1f MB\n", argv[2], im.entries.size(), ru.ru_maxrss/1024.0);
    return 0;
}

static int cmd_count(int argc, char ** argv) {
    llama_hessian h;
    if (!h.open(argv[2]) || !h.has(argv[3])) {
        fprintf(stderr, "count: no Gram for %s in %s\n", argv[3], argv[2]);
        return 1;
    }
    const double n = h.count(argv[3]);
    printf("count %s: %.0f\n", argv[3], n);
    return argc > 4 && n != atof(argv[4]) ? 1 : 0;
}

int main(int argc, char ** argv) {
    const std::string c = argc > 1 ? argv[1] : "";
    if (c == "count" && argc >= 4) {
        return cmd_count(argc, argv);
    }
    if (c == "imatrix-load" && argc >= 3) {
        return cmd_imatrix_load(argv);
    }
    if (c == "check" && argc >= 3) {
        return cmd_check(argc, argv);
    }
    if (c == "compare" && argc >= 4) {
        return cmd_compare(argc, argv);
    }
    if (c == "sum" && argc >= 5) {
        return cmd_sum(argc, argv);
    }
    if (c == "diag" && argc >= 4) {
        return cmd_diag(argc, argv);
    }
    fprintf(stderr, "usage: %s check FILE [--weights N] | compare A B [--tol T] [--per-count] | sum A B C [--tol T] | diag FILE IMATRIX [--tol T] | imatrix-load FILE\n", argv[0]);
    return 1;
}
