# Imports from 3rd party libraries
import numpy as np
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt

from src.models.train_model import ModelTraining


def generate_report(
    model_train_obj: ModelTraining,
    output_pdf: str,
    scaler_df: pd.DataFrame,
    metadata_path: str,
    figures_filepath: str,
    logo_filepath: str,
    xlim=None,
    ylim=True,
):
    """
    Save multiple plots to a PDF file, organized in a 4x4 matrix on each page.

    Parameters:
    - datas: List of data for each plot.
    - output_pdf: Name of the output PDF file.
    """
    simulation_dict, train_test_dict = model_train_obj.get_model_data_dict(data_agg="train")
    dict_keys = list(simulation_dict.keys())
    COLS = 2
    ROWS = 4
    ppg = ROWS * COLS
    now = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    # Plotting the table and removing all axes
    fig_table, ax = plt.subplots(
        figsize=(6, 2)
    )  # set the size that you'd like (width, height)
    ax.axis("off")

    tbl = ax.table(
        cellText=scaler_df.values,
        colLabels=list(scaler_df.columns),
        cellLoc="center",
        loc="center",
    )

    # Create the table and scale it to fit the fig
    for (i, j), cell in tbl.get_celld().items():
        if i == 0:  # header cells
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#F25D18")

    d = {}
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            data = line.split(sep=":")
            d[data[0]] = data[1]

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(20)
    tbl.scale(2, 4)  # may need to adjust this for your data
    plt.savefig(rf"{figures_filepath}\table_image.png", dpi=200, bbox_inches="tight")
    plt.close(fig_table)

    pdf = FPDF(format="A4")  # A4 (210 by 297 mm)
    pdf.add_page()
    pdf.set_font("helvetica", "B", 8)
    pdf.image(logo_filepath, w=30, h=10, x=170, y=10)
    pdf.image(logo_filepath, w=30, h=10, x=10, y=277)

    # Set the title of the document
    pdf.set_font("helvetica", "B", 24)
    pdf.set_text_color(r=242, g=93, b=24)
    pdf.ln(15)
    pdf.write(5, "BDSD State Space Model Report")

    # Specify the reference number of document
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(r=242, g=93, b=24)
    pdf.ln(15)
    pdf.write(5, f"IDBS Reference: {d['IDBS']}")

    # Write other data to the front cover
    pdf.set_font("helvetica", "", 16)
    pdf.set_text_color(r=242, g=93, b=24)
    pdf.set_text_color(r=0, g=0, b=0)
    pdf.ln(13)
    pdf.write(7, f"Report Generated for {d['Asset']}")
    pdf.ln(2)
    pdf.write(7, f"Dataset for Model Training: {d['Dataset']}")
    pdf.ln(2)
    pdf.write(7, f"States in Model: {(', ').join(model_train_obj.states)}")
    pdf.ln(6)
    pdf.write(7, f"Inputs in Model: {(', ').join(model_train_obj.inputs)}")

    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(r=242, g=93, b=24)
    pdf.ln(15)
    pdf.write(7, f"Table of Scaler Values from {d['scaler_type']}")

    pdf.image(rf"{figures_filepath}\table_image.png", w=160, h=100, x=25, y=120)

    pdf.set_font("helvetica", "I", 16)
    pdf.set_text_color(r=0, g=0, b=0)
    pdf.ln(132)
    pdf.write(7, f"Report Author: {d['Author']}")
    pdf.ln(2)
    pdf.write(7, f"GitHub Link: {d['Github']}")
    pdf.ln(2)
    pdf.write(7, f"This report was generated on {now}")

    pdf.add_page()
    pdf.set_font("helvetica", "B", 8)
    pdf.image(logo_filepath, w=30, h=10, x=170, y=10)
    pdf.image(logo_filepath, w=30, h=10, x=10, y=277)

    # Set the title of the document
    pdf.set_font("helvetica", "B", 24)
    pdf.set_text_color(r=242, g=93, b=24)
    pdf.ln(127)
    pdf.write(5, "Model Training Dataset")
    df_sim_concat = pd.concat(simulation_dict.values(), ignore_index=True)
    df_train_concat = pd.concat(train_test_dict.values(), ignore_index=True)
    for test_label in model_train_obj.states:
        pdf.add_page()
        pdf.image(logo_filepath, w=30, h=10, x=170, y=10)
        pdf.image(logo_filepath, w=30, h=10, x=10, y=277)
        # Set the title of the document
        pdf.set_font("helvetica", "B", 24)
        pdf.set_text_color(r=242, g=93, b=24)
        pdf.ln(5)
        pdf.write(5, f"State: {test_label}")
        if df_sim_concat[test_label].max() > df_train_concat[test_label].max():
            max_value = df_sim_concat[test_label].max()
        else:
            max_value = df_train_concat[test_label].max()

        if df_sim_concat[test_label].min() < df_sim_concat[test_label].min():
            min_value = df_sim_concat[test_label].min()
        else:
            min_value = df_sim_concat[test_label].min()

        for i in range(0, len(simulation_dict.keys()), ppg):
            if i != 0:
                pdf.add_page()
                pdf.image(logo_filepath, w=30, h=10, x=170, y=10)
                pdf.image(logo_filepath, w=30, h=10, x=10, y=277)
            fig, axs = plt.subplots(ROWS, COLS, figsize=(8, 10), squeeze=False)
            fig.subplots_adjust(top=0.8)

            for count, ax_test in enumerate(axs.reshape(-1)):
                if count + i < len(simulation_dict.keys()):
                    key = dict_keys[count + i]
                    time = np.arange(0, len(simulation_dict[key][test_label]), 1)
                    ax_test.plot(
                        time,
                        simulation_dict[key][test_label],
                        "ro-",
                        label="Simulated Data",
                        markersize=3.5,
                    )
                    ax_test.plot(
                        time,
                        train_test_dict[key][test_label],
                        "bo-",
                        label="Experimental Data",
                        markersize=3.5,
                    )
                    ax_test.set_title(key, size="medium", weight="bold")
                    ax_test.grid()
                    if ylim:
                        if min_value > 200:
                            ax_test.set_ylim(
                                min_value - (min_value * 0.2),
                                max_value + (max_value * 0.2),
                            )
                        else:
                            ax_test.set_ylim(0, max_value + (max_value * 0.2))
                    if xlim is not None:
                        ax_test.set_xlim(-1.5, xlim)

            # If on the last page and there are fewer than 12 plots, remove extra subplots
            if len(simulation_dict.keys()) - i < ppg:
                for j in range(len(simulation_dict.keys()) - i, len(axs.flatten())):
                    axs.ravel()[j].remove()

            axs[ROWS - 1][COLS - 1].legend()
            # if ppg < len(simulation_dict.keys()) - i:
            #     axs[ROWS - 1][COLS - 1].legend()
            # else:
            #     axs[math.ceil((len(simulation_dict.keys()) - i)/COLS) - 1][COLS - 1].legend()
            # fig.suptitle("Training Data Set", size= "x-large", weight= "bold", y=0.98)
            fig.supxlabel("Day", size="x-large", weight="bold")
            fig.supylabel(f"{test_label}", size="x-large", weight="bold")
            fig.tight_layout()

            plt.savefig(rf"{figures_filepath}\{test_label}_{i}.png", dpi=200)
            plt.close(fig)
            pdf.image(
                rf"{figures_filepath}\{test_label}_{i}.png",
                w=190,
                h=250,
                x=10,
                y=22,
            )

    pdf.output(output_pdf)
