import matplotlib.pyplot as plt
import numpy as np

from model_utils import interaction_coef, interaction_se


def plot_coefs(results, delta=0.3, palette=None, labels=None, ylabels=True):
    if labels is None:
        labels = ['FN', 'LN', 'FLN', 'A']
    if palette is None:
        palette = ['#df2935', '#3772ff', '#fdca40', 'lightgrey']
    init = -1.5 * delta
    for i, res in enumerate(results):
        offset = init + i * delta
        means = res.params[1:]
        CIs = res.conf_int(alpha=0.05)
        lowers = CIs[0][1:]
        uppers = CIs[1][1:]
        for j, (m, l, u) in enumerate(zip(means, lowers, uppers)):
            j *= 4
            edgecolors = palette[i]
            facecolors = 'white' if l < 0 < u else palette[i]
            plt.scatter(offset + j, m, edgecolors=edgecolors, facecolors=facecolors, alpha=1, zorder=10, label=labels[i] if facecolors != 'white' else None)
            plt.plot([offset + j, offset + j], [l, u], color=palette[i], zorder=5)
    if ylabels:
        plt.xticks(range(0, len(means) * 4, 4), labels=['$D_{sport}$', '$D_{art}$', '$D_{other}$', '$N_{India}$', '$N_{UK}$', '$N_{other}$',
                                                        '$G_s$', '$G_m$', '$a_s$',  '$a_m$',
                                                        '$r_s$',  '$r_m$', '$f_{FN}$', '$f_{LN}$',
                                                        '$l_{FN}$', '$l_{LN}$', '$t$'], fontsize=20)

    else:
        plt.xticks(range(0, len(means) * 4, 4), labels=[])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(list(by_label.values())[::-1], list(by_label.keys())[::-1], fontsize=18)
    plt.yticks(fontsize=20)
    plt.axhline(0, linestyle='dotted', color='black')
    plt.ylim(-1.5, 1.5)


def plot_model_coefs(coefs, models, palette, labels, delta=0.1):
    k = 1
    for coef in coefs:
        offset = k - delta
        for res, color, label in zip(models, palette, labels):
            params = interaction_coef(coef, res.params)
            se = interaction_se(coef, res.cov_params())
            lower = params - 1.96 * se
            upper = params + 1.96 * se
            plt.scatter(offset, params, color=color, label=label, zorder=10)
            plt.plot([offset, offset], [lower, upper], zorder=5)
            offset += delta
        k += 1

def plot_gender(models):
    ff_coef = [
        "Intercept",
        "C(source_female)[T.True]",
        "C(target_female)[T.True]",
        "C(source_female)[T.True]:C(target_female)[T.True]",
    ]
    mf_coef = ["Intercept", "C(target_female)[T.True]"]
    fm_coef = ["Intercept", "C(source_female)[T.True]"]
    mm_coef = ["Intercept"]

    coefs = [ff_coef, mf_coef, fm_coef, mm_coef]
    palette = ["#df2935", "#3772ff", "#fdca40", "lightgrey"]

    labels_i = ["FF", "MF", "FM", "MM"]
    labels = ["FN", "LN", "FLN", "A"]

    plot_coefs(coefs, models, palette, labels)

    plt.xticks(range(1, 5), labels_i)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(
        list(by_label.values()),
        list(by_label.keys()),
        fontsize=16,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    plt.ylabel("Average log odds", fontsize=20)
    plt.xlabel("Interaction", fontsize=20)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylim(-3.1, 0.5)


def plot_gender_occupations(models):
    domains = ['Politics', 'Sport', 'Art', 'Other']
    ff_coef_domain = [i for i in models[0].params.index if
                      'sport' in i or 'art' in i or 'Intercept' in i or 'female' in i or 'O_Other' in i]
    mf_coef_domain = [i for i in ff_coef_domain if 'source_female' not in i]
    fm_coef_domain = [i for i in ff_coef_domain if 'target_female' not in i]
    mm_coef_domain = [i for i in ff_coef_domain if 'female' not in i]

    def has_any(i, domains):
        for j in domains:
            if j.lower() in i:
                return True
        return False

    m = 1
    for domain in domains[:-1]:
        other_domains = [i for i in domains if i != domain]
        ff_coef = [i for i in ff_coef_domain if not has_any(i, other_domains)]
        mf_coef = [i for i in mf_coef_domain if not has_any(i, other_domains)]
        fm_coef = [i for i in fm_coef_domain if not has_any(i, other_domains)]
        mm_coef = [i for i in mm_coef_domain if not has_any(i, other_domains)]
        coefs = [ff_coef, mf_coef, fm_coef, mm_coef]
        labels_i = ['FF', 'MF', 'FM', 'MM']

        palette = ['#df2935', '#3772ff', '#fdca40', 'lightgrey']
        labels = ['FN', 'LN', 'FLN', 'A']
        plt.subplot(1, 3, m)
        plot_model_coefs(coefs, models, palette, labels)
        plt.xticks(range(1, 5), labels_i)

        if m == 1:
            plt.ylabel('Average log odds', fontsize=20)
            plt.yticks(fontsize=16)
        else:
            plt.yticks(np.arange(-3, 1, 0.5), labels=[], fontsize=16)

        if m == 2:
            plt.xlabel('Interaction', fontsize=20)

        m += 1
        plt.xticks(fontsize=16)

        plt.ylim(-3, 0.5)
        plt.title(domain, fontsize=20)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(list(by_label.values()), list(by_label.keys()), fontsize=16, loc='center left',
               bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(wspace=0.05, hspace=0)

def plot_party(models):
    rr_coef = ['Intercept', 'C(source_US_party)[T.Republican]', 'C(target_US_party)[T.Republican]', 'C(source_US_party)[T.Republican]:C(target_US_party)[T.Republican]']
    dr_coef = ['Intercept', 'C(target_US_party)[T.Republican]']
    rd_coef = ['Intercept', 'C(source_US_party)[T.Republican]']
    dd_coef = ['Intercept']

    coefs = [rr_coef, dd_coef, dr_coef, rd_coef]
    labels_i = ['RR', 'DD', 'DR', 'RD']

    palette = ['#df2935', '#3772ff', '#fdca40', 'lightgrey']
    labels = ['FN', 'LN', 'FLN', 'A']

    plt.subplot(1,2,1)
    plot_model_coefs(coefs, models, palette, labels)

    plt.xticks(range(1, 5), labels_i)
    plt.ylabel('Average log odds', fontsize=20)
    plt.xlabel('Interaction', fontsize=20)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylim(-3.1, 0.5)
    plt.subplot(1, 2, 2)

    cc_coef = ['Intercept', 'C(source_congress)[T.True]', 'C(target_congress)[T.True]', 'C(source_congress)[T.True]:C(target_congress)[T.True]']
    nc_coef = ['Intercept', 'C(target_congress)[T.True]']
    cn_coef = ['Intercept', 'C(source_congress)[T.True]']
    nn_coef = ['Intercept']

    coefs = [cc_coef, nc_coef, cn_coef, nn_coef]
    labels_i = ['CC', 'NC', 'CN', 'NN']
    plot_model_coefs(coefs, models, palette, labels_i)

    plt.xticks(range(1, 5), labels_i)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(list(by_label.values()), list(by_label.keys()), fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Interaction', fontsize=20)
    plt.yticks(np.arange(-3, 1, 0.5), labels=[], fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylim(-3.1, 0.5)
    plt.subplots_adjust(wspace=0.05, hspace=0)