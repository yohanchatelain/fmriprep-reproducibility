import matplotlib.pyplot as plt
import matplotlib as mpl


def main(vmin=0, vmax=12):
    fig, ax = plt.subplots(figsize=(10, 0.75))
    fig.subplots_adjust(bottom=0.4, left=0.01, right=0.99, top=0.9)

    cmap = mpl.cm.RdYlGn
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="horizontal",
    )

    fig.savefig("colorbar_sigbit.pdf")


if __name__ == "__main__":
    main()
