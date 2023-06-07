import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px


def main(vmin=0, vmax=12):
    fig, ax = plt.subplots(figsize=(10, 0.75))
    fig.subplots_adjust(bottom=0.4, left=0.01, right=0.99, top=0.9)

    cmap = mpl.colormaps.get_cmap("RdYlGn")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="horizontal",
    )

    fig.savefig("colorbar_sigbit.pdf")

    cmap = cmap.resampled(12)

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="horizontal",
    )

    fig.savefig("colorbar_sigbit_discrete.pdf")


if __name__ == "__main__":
    main()
