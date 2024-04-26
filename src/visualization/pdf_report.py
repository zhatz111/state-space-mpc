"""Main code for simulating closed-loop MPC
    Created by Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
    Created: 2024-04-24
    Modified: 2024-04-24
"""

# Standard Library Imports
from typing import Union
from pathlib import Path
from datetime import datetime

# Imports from 3rd party libraries
import numpy as np
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt

# Repository Imports
from src.models.train_model import ModelTraining

print(Path().absolute())

def generate_report(
    model_train_obj: ModelTraining,
    output_folder: Union[Path, str],
    metadata: dict,
    xlim=None,
    ylim=True,
):
    """
    Save multiple plots to a PDF file, organized in a 4x4 matrix on each page.

    Parameters:
    - datas: List of data for each plot.
    - output_pdf: Name of the output PDF file.
    """
    output_path = Path(output_folder, f"{metadata['Training Data Study']}_report.pdf")
    figures_filepath = Path(output_folder, "report_figures")
    figures_filepath.mkdir(parents=True, exist_ok=True)
    logo_filepath = str(Path(Path().absolute(), "reports", "report_info", "GSK_logo_2022.png"))

    simulation_train_dict, train_dict = model_train_obj.get_model_data_dict(data_agg="train")
    # simulation_test_dict, test_dict = model_train_obj.get_model_data_dict(data_agg="test")

    dict_keys = list(simulation_train_dict.keys())
    cols = 2
    rows = 4
    ppg = rows * cols # subplot dimensions
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    # Plotting the table and removing all axes
    fig_table, ax = plt.subplots(
        figsize=(6, 2)
    )  # set the size that you'd like (width, height)
    ax.axis("off")

    scaler_dict = {}
    for key, value in metadata["scaler"].items():
        if isinstance(value,list) and len(value) == len(metadata["scaler"]["feature_names_in_"]):
            scaler_dict[key] = value
    scaler_df = pd.DataFrame.from_dict(scaler_dict).round(3)

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

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14)
    tbl.scale(2, 4)  # may need to adjust this for your data
    plt.savefig(rf"{str(figures_filepath)}\scaler_table.png", dpi=200, bbox_inches="tight")
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
    pdf.write(5, "BDSD MPC Training Report")

    # Specify the reference number of document
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(r=242, g=93, b=24)
    pdf.ln(15)
    pdf.write(5, f"IDBS Reference: {metadata['IDBS Number']}")

    # Write other data to the front cover
    pdf.set_font("helvetica", "", 16)
    pdf.set_text_color(r=242, g=93, b=24)
    pdf.set_text_color(r=0, g=0, b=0)
    pdf.ln(13)
    pdf.write(7, f"Report Generated for {metadata['Asset']}")
    pdf.ln(2)
    pdf.write(7, f"Dataset for Model Training: {metadata['Training Data Study']}")
    pdf.ln(2)
    pdf.write(7, f"States in Model: {(', ').join(model_train_obj.states)}")
    pdf.ln(6)
    pdf.write(7, f"Inputs in Model: {(', ').join(model_train_obj.inputs)}")

    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(r=242, g=93, b=24)
    pdf.ln(15)
    pdf.write(7, "Table of Scaler Values from MinMaxScaler")

    pdf.image(rf"{str(figures_filepath)}\scaler_table.png", w=160, h=100, x=25, y=120)

    pdf.set_font("helvetica", "I", 16)
    pdf.set_text_color(r=0, g=0, b=0)
    pdf.ln(132)
    pdf.write(7, f"Report Author: {metadata['Operator']}")
    pdf.ln(2)
    pdf.write(7, f"GitHub Link: {metadata['Github Link']}")
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
    df_sim_concat = pd.concat(simulation_train_dict.values(), ignore_index=True)
    df_train_concat = pd.concat(train_dict.values(), ignore_index=True)
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

        for i in range(0, len(simulation_train_dict.keys()), ppg):
            if i != 0:
                pdf.add_page()
                pdf.image(logo_filepath, w=30, h=10, x=170, y=10)
                pdf.image(logo_filepath, w=30, h=10, x=10, y=277)
            fig, axs = plt.subplots(rows, cols, figsize=(8, 10), squeeze=False)
            fig.subplots_adjust(top=0.8)

            for count, ax_test in enumerate(axs.reshape(-1)):
                if count + i < len(simulation_train_dict.keys()):
                    key = dict_keys[count + i]
                    time = np.arange(0, len(simulation_train_dict[key][test_label]), 1)
                    ax_test.plot(
                        time,
                        simulation_train_dict[key][test_label],
                        "ro-",
                        label="Simulated Data",
                        markersize=3.5,
                    )
                    ax_test.plot(
                        time,
                        train_dict[key][test_label],
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
            if len(simulation_train_dict.keys()) - i < ppg:
                for j in range(len(simulation_train_dict.keys()) - i, len(axs.flatten())):
                    axs.ravel()[j].remove()

            axs[rows - 1][cols - 1].legend()
            # if ppg < len(simulation_dict.keys()) - i:
            #     axs[ROWS - 1][COLS - 1].legend()
            # else:
            #     axs[math.ceil((len(simulation_dict.keys()) - i)/COLS) - 1][COLS - 1].legend()
            # fig.suptitle("Training Data Set", size= "x-large", weight= "bold", y=0.98)
            fig.supxlabel("Day", size="x-large", weight="bold")
            fig.supylabel(f"{test_label}", size="x-large", weight="bold")
            fig.tight_layout()

            plt.savefig(rf"{str(figures_filepath)}\{test_label}_{i}.png", dpi=200)
            plt.close(fig)
            pdf.image(
                rf"{str(figures_filepath)}\{test_label}_{i}.png",
                w=190,
                h=250,
                x=10,
                y=22,
            )

    pdf.output(str(output_path))
