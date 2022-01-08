import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.colors as mcolors


def plot_shap_amino(
    shap_values,
    seq_len,
    seq_name="",
    vmin=-0.01,
    vmax=0.01,
    max_bp=500,
    save_path=None,
    trimp_shap=True,
    dpi=500,
):
    """Plot explanation results
    """

    num_pic = 1 if seq_len <= (max_bp * 3) else 2
    seq_len = seq_len // 3

    if trimp_shap:
        shap_values[(seq_len * 3) + 3 :] = 0

    h_fig_size = {1: 2, 2: 4, 3: 6, 4: 9, 5: 11, 6: 14}
    fig, ax = plt.subplots(
        nrows=num_pic * 3, ncols=1, figsize=(20, h_fig_size[num_pic * 3])
    )
    plt.subplots_adjust(
        left=0, bottom=0, right=0.5, top=1 * 1.0, wspace=0.2, hspace=0.1
    )
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)

    count = 0
    for i in range(3):
        frames = (
            np.concatenate(
                (shap_values[i:].sum(axis=1), np.array([0.0, 0.0, 0.0, 0.0] * i))
            )
            .reshape(-1, 3)
            .sum(axis=1)
        )

        arr = []
        for n in range(num_pic):
            arr.append(np.array([frames[n * max_bp : max_bp * (n + 1)].tolist(),] * 75))

        for j in range(num_pic):
            axs = ax[count]
            img = arr[j]

            im = axs.imshow(img, cmap=plt.cm.RdBu, norm=norm)

            axs.xaxis.set_ticks(np.arange(0, max_bp + 1, 50))
            axs.xaxis.set_minor_locator(tck.AutoMinorLocator())
            axs.set_xticklabels(
                [x + (max_bp * j) for x in np.arange(0, max_bp + 1, 50)]
            )
            axs.yaxis.set_major_locator(plt.NullLocator())
            axs.set_ylabel("SHAP value")

            # Add Red line
            if j == num_pic - 1:
                lx = seq_len % max_bp if seq_len % max_bp != 0 else max_bp
                axs.axvline(x=lx, color="r", linestyle="--")

            plt.colorbar(im, ax=axs, fraction=0.008, pad=0.04)

            if count == 0:
                axs.set_title(
                    f"{seq_name}\n Amino Sequnces length: {seq_len}\n\n Frame: {i+1} \n{max_bp*j} - {max_bp*(j+1)}"
                )
            else:
                axs.set_title(f"Frame: {i+1} \n{max_bp*j} - {max_bp*(j+1)}")
            count += 1

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    else:
        plt.show()
    plt.cla()
    plt.clf()
    plt.close()
