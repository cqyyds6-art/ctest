import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

# 强制使用独立窗口后端
matplotlib.use("TkAgg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from scipy.signal import find_peaks
from tkinter import filedialog, messagebox, simpledialog, ttk


@dataclass(slots=True)
class SinglePointOptions:
    normalize: bool = True
    combine_with_photo: bool = True
    save_plot: bool = True
    peak_labeling: bool = False
    graphene_calc: bool = False


@dataclass(slots=True)
class MappingOptions:
    split_txt: bool = True
    generate_overlay: bool = True
    normalize: bool = True
    generate_individual_plot: bool = False
    peak_labeling: bool = False
    graphene_calc: bool = False


class RamanProcessorApp:
    """Raman 数据处理桌面应用（重构版）。"""

    IMAGE_EXTENSIONS = [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Raman Processor v6.0 (Refactored)")
        self.center_window(self.root, 450, 420)
        self._init_plot_theme()
        self._build_main_ui()

    @staticmethod
    def _init_plot_theme() -> None:
        plt.rcParams["font.sans-serif"] = ["Arial"]
        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["mathtext.rm"] = "Arial"
        plt.rcParams["axes.linewidth"] = 1.5

    def _build_main_ui(self) -> None:
        tk.Label(self.root, text="Raman Workspace", font=("Arial", 16, "bold"), pady=10).pack()

        tips = (
            "💡 Tip: 'Mode A' handles Batch Single-Points.\n"
            "'Mode B' handles large Mapping files.\n"
            "Smart Detector routes you to settings."
        )
        tk.Label(self.root, text=tips, font=("Arial", 9), fg="#666", justify=tk.LEFT).pack(pady=5)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=10)

        tk.Button(
            btn_frame,
            text="🚀 SMART AUTO-DETECTOR",
            command=self.smart_detector,
            font=("Arial", 11, "bold"),
            height=2,
            bg="#c8e6c9",
            fg="#2e7d32",
        ).pack(fill=tk.X, pady=10)

        tk.Canvas(btn_frame, height=1, relief="sunken", bg="gray").pack(fill="x", pady=10)

        tk.Button(
            btn_frame,
            text="Mode A: Single Point (Batch Support)",
            command=self.setup_single_point,
            font=("Arial", 10),
            height=2,
            bg="#e1f5fe",
        ).pack(fill=tk.X, pady=5)

        tk.Button(
            btn_frame,
            text="Mode B: Mapping (Manual)",
            command=self.setup_mapping,
            font=("Arial", 10),
            height=2,
            bg="#fff3e0",
        ).pack(fill=tk.X, pady=5)

    @staticmethod
    def center_window(win: tk.Tk | tk.Toplevel, width: int, height: int) -> None:
        win.update_idletasks()
        ws, hs = win.winfo_screenwidth(), win.winfo_screenheight()
        x_pos = (ws // 2) - (width // 2)
        y_pos = (hs // 2) - (height // 2)
        win.geometry(f"{width}x{height}+{x_pos}+{y_pos}")

    @staticmethod
    def _find_related_image(base_path: Path) -> Path | None:
        for ext in RamanProcessorApp.IMAGE_EXTENSIONS:
            candidate = base_path.with_suffix(ext)
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        minimum = float(series.min())
        maximum = float(series.max())
        if maximum == minimum:
            return pd.Series([0.0] * len(series), index=series.index)
        return (series - minimum) / (maximum - minimum) * 100

    def apply_plot_style(
        self,
        ax: plt.Axes,
        x_data: pd.Series,
        y_data: pd.Series,
        is_norm: bool,
        do_peaks: bool = False,
        do_graphene: bool = False,
    ) -> None:
        ax.plot(x_data, y_data, color="blue", lw=1.2)
        ax.set_xlim(float(x_data.min()), float(x_data.max()))

        ymin, ymax = float(y_data.min()), float(y_data.max())
        yrange = max(ymax - ymin, 1e-9)

        if do_peaks:
            ax.text(
                0.98,
                0.96,
                "PEAK DETECT ON",
                transform=ax.transAxes,
                fontsize=8,
                color="orange",
                fontweight="bold",
                ha="right",
                va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="orange"),
            )

            peaks, _ = find_peaks(y_data, prominence=yrange * 0.1, distance=10, width=2)
            peak_info: dict[str, float] = {}

            for peak_idx in peaks:
                px = float(x_data.iloc[peak_idx])
                py = float(y_data.iloc[peak_idx])
                ax.annotate(
                    f"{int(px)}\n({py:.1f})",
                    xy=(px, py),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="#D32F2F",
                    fontweight="bold",
                )
                ax.plot(px, py, "r.", markersize=4)

                if 1300 < px < 1400:
                    peak_info["D"] = py
                elif 1540 < px < 1620:
                    peak_info["G"] = py
                elif 2600 < px < 2800:
                    peak_info["2D"] = py

            if do_graphene and "G" in peak_info:
                g = peak_info["G"]
                results = []
                if "D" in peak_info:
                    results.append(f"ID/IG: {peak_info['D'] / g:.3f}")
                if "2D" in peak_info:
                    results.append(f"I2D/IG: {peak_info['2D'] / g:.3f}")
                if results:
                    ax.text(
                        0.05,
                        0.95,
                        "\n".join(results),
                        transform=ax.transAxes,
                        fontsize=10,
                        color="green",
                        fontweight="bold",
                        va="top",
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="green"),
                    )

        if is_norm:
            ax.set_ylim(-5, 130 if do_peaks else 105)
            y_label = "Normalized Intensity (a.u.)"
        else:
            top_margin = 0.35 if do_peaks else 0.1
            ax.set_ylim(ymin - 0.05 * yrange, ymax + top_margin * yrange)
            y_label = "Intensity (a.u.)"

        ax.tick_params(axis="both", which="major", direction="out", labelsize=11, width=1.5, length=6)
        ax.set_xlabel(r"Wavenumber ($\rm{cm^{-1}}$)", fontsize=14, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
        for spine in ["top", "bottom", "left", "right"]:
            ax.spines[spine].set_linewidth(1.5)

    def smart_detector(self) -> None:
        fpath = filedialog.askopenfilename(title="Select Raman File (.txt)", filetypes=[("Text", "*.txt")])
        if not fpath:
            return

        with open(fpath, "r", encoding="utf-8") as file:
            first_line = file.readline().strip().split("\t")

        if len(first_line) <= 3:
            messagebox.showinfo("Smart Detector", "Detected: [SINGLE POINT]")
            self.setup_single_point(pre_files=[fpath])
        else:
            messagebox.showinfo("Smart Detector", f"Detected: [MAPPING]\nColumns: {len(first_line)}")
            self.setup_mapping(pre_file=fpath)

    def setup_single_point(self, pre_files: list[str] | None = None) -> None:
        config_win = tk.Toplevel(self.root)
        config_win.title("Single Point Batch Settings")
        self.center_window(config_win, 400, 350)
        config_win.attributes("-topmost", True)

        tk.Label(config_win, text="Batch Processing Configuration", font=("Arial", 10, "bold")).pack(pady=10)

        options = SinglePointOptions()
        v_norm = tk.BooleanVar(value=options.normalize)
        v_comb = tk.BooleanVar(value=options.combine_with_photo)
        v_plot = tk.BooleanVar(value=options.save_plot)
        v_peaks = tk.BooleanVar(value=options.peak_labeling)
        v_graph = tk.BooleanVar(value=options.graphene_calc)

        tk.Checkbutton(config_win, text="1. Normalize Intensity (0-100)", variable=v_norm).pack(anchor="w", padx=60)
        tk.Checkbutton(config_win, text="2. Auto-combine with Photo", variable=v_comb).pack(anchor="w", padx=60)
        tk.Checkbutton(config_win, text="3. Save High-Res Spectrum Plot", variable=v_plot).pack(anchor="w", padx=60)
        tk.Label(config_win, text="Advanced Analysis:", fg="#666").pack(anchor="w", padx=60, pady=(10, 2))
        tk.Checkbutton(config_win, text="✨ Peak Labeling", variable=v_peaks, fg="#D32F2F").pack(anchor="w", padx=60)
        tk.Checkbutton(config_win, text="   └─ Graphene Calc (ID/IG)", variable=v_graph, fg="green").pack(anchor="w", padx=80)

        def start() -> None:
            files = pre_files if pre_files else filedialog.askopenfilenames(title="Select Files", filetypes=[("Text", "*.txt")])
            if not files:
                return

            config_win.destroy()
            self.process_single_point(
                list(files),
                SinglePointOptions(
                    normalize=v_norm.get(),
                    combine_with_photo=v_comb.get(),
                    save_plot=v_plot.get(),
                    peak_labeling=v_peaks.get(),
                    graphene_calc=v_graph.get(),
                ),
            )

        tk.Button(config_win, text="Confirm & Execute", command=start, bg="#b9f6ca", height=2, width=20).pack(pady=15)

    def process_single_point(self, files: Iterable[str], options: SinglePointOptions) -> None:
        files = list(files)
        prog_win = tk.Toplevel(self.root)
        self.center_window(prog_win, 400, 100)
        pb = ttk.Progressbar(prog_win, length=300, maximum=len(files))
        pb.pack(pady=20)

        skipped: list[str] = []
        processed_count = 0

        for i, fpath in enumerate(files):
            try:
                with open(fpath, "r", encoding="utf-8") as file:
                    header_cols = len(file.readline().strip().split("\t"))
                if header_cols > 3:
                    skipped.append(Path(fpath).name)
                    continue

                data = pd.read_csv(fpath, sep="\t", header=None, names=["W", "I"])
                plot_y = self._normalize(data["I"]) if options.normalize else data["I"]
                base = Path(fpath).with_suffix("")

                if options.save_plot:
                    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
                    self.apply_plot_style(ax, data["W"], plot_y, options.normalize, options.peak_labeling, options.graphene_calc)
                    fig.savefig(str(base) + "_Plot.png", bbox_inches="tight")
                    plt.close(fig)

                if options.combine_with_photo:
                    image_path = self._find_related_image(Path(fpath))
                    if image_path:
                        fig_c, (ax_i, ax_s) = plt.subplots(2, 1, figsize=(8, 10), dpi=200)
                        ax_i.imshow(mpimg.imread(image_path))
                        ax_i.axis("off")
                        self.apply_plot_style(ax_s, data["W"], plot_y, options.normalize, options.peak_labeling, options.graphene_calc)
                        fig_c.savefig(str(base) + "_Combined.png", bbox_inches="tight")
                        plt.close(fig_c)

                processed_count += 1
            except Exception as exc:  # noqa: BLE001
                print(f"Error skipping {fpath}: {exc}")
            finally:
                pb["value"] = i + 1
                prog_win.update()

        prog_win.destroy()

        report = f"Successfully processed {processed_count} files."
        if skipped:
            report += f"\n\nSkipped {len(skipped)} Mapping files (use Mode B instead):\n" + "\n".join(skipped[:5])
        messagebox.showinfo("Batch Result", report)

    def setup_mapping(self, pre_file: str | None = None) -> None:
        config_win = tk.Toplevel(self.root)
        config_win.title("Mapping Configuration")
        self.center_window(config_win, 400, 380)

        options = MappingOptions()
        v_split = tk.BooleanVar(value=options.split_txt)
        v_overlay = tk.BooleanVar(value=options.generate_overlay)
        v_norm = tk.BooleanVar(value=options.normalize)
        v_indiv = tk.BooleanVar(value=options.generate_individual_plot)
        v_peaks = tk.BooleanVar(value=options.peak_labeling)
        v_graph = tk.BooleanVar(value=options.graphene_calc)

        tk.Checkbutton(config_win, text="1. Split & Save .txt Files (Raw Data)", variable=v_split).pack(anchor="w", padx=60, pady=2)
        tk.Checkbutton(config_win, text="   └> Also Generate Plot for each .txt?", variable=v_indiv).pack(anchor="w", padx=80)
        tk.Checkbutton(config_win, text="2. Generate Mapping Overlay Images", variable=v_overlay).pack(anchor="w", padx=60, pady=2)
        tk.Checkbutton(config_win, text="3. Normalize Spectrum in Plots?", variable=v_norm).pack(anchor="w", padx=60, pady=2)
        tk.Label(config_win, text="Advanced Analysis:", fg="#666").pack(anchor="w", padx=60, pady=(10, 2))
        tk.Checkbutton(config_win, text="✨ Peak Labeling", variable=v_peaks, fg="#D32F2F").pack(anchor="w", padx=60)
        tk.Checkbutton(config_win, text="   └─ Graphene Calc (ID/IG)", variable=v_graph, fg="green").pack(anchor="w", padx=80)

        def start() -> None:
            map_file = pre_file if pre_file else filedialog.askopenfilename(title="Select Mapping .txt", filetypes=[("Text", "*.txt")])
            if not map_file:
                return

            config_win.destroy()
            self.process_mapping(
                map_file,
                MappingOptions(
                    split_txt=v_split.get(),
                    generate_overlay=v_overlay.get(),
                    normalize=v_norm.get(),
                    generate_individual_plot=v_indiv.get(),
                    peak_labeling=v_peaks.get(),
                    graphene_calc=v_graph.get(),
                ),
            )

        tk.Button(config_win, text="Confirm & Run", command=start, bg="#ffcc80", height=2, width=20).pack(pady=20)

    def calibrate_image(self, img_path: str) -> tuple[float, float, float, float] | None:
        img = mpimg.imread(img_path)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(img)

        prompts = ["X-Axis Left", "X-Axis Right", "Y-Axis Bottom", "Y-Axis Top"]
        pix, vals = [], []
        for i, prompt in enumerate(prompts, start=1):
            plt.title(f"STEP {i}/4: Click on {prompt} scale", color="red", fontweight="bold")
            plt.draw()
            pts = plt.ginput(1, timeout=0)
            if not pts:
                plt.close(fig)
                return None

            ax.plot(pts[0][0], pts[0][1], "rx", markersize=15)
            plt.draw()

            temp = tk.Toplevel(self.root)
            temp.withdraw()
            value = simpledialog.askfloat("Calibration", f"Value for {prompt}:", parent=temp)
            temp.destroy()

            if value is None:
                plt.close(fig)
                return None

            pix.append(pts[0])
            vals.append(value)

        plt.close(fig)
        s_x = (pix[1][0] - pix[0][0]) / (vals[1] - vals[0])
        o_x = pix[0][0] - s_x * vals[0]
        s_y = (pix[3][1] - pix[2][1]) / (vals[3] - vals[2])
        o_y = pix[2][1] - s_y * vals[2]
        return s_x, o_x, s_y, o_y

    def process_mapping(self, map_file: str, options: MappingOptions) -> None:
        img_path: str | None = None
        calibration = None

        if options.generate_overlay:
            base = Path(map_file).with_suffix("")
            candidate = self._find_related_image(base)
            if candidate is None:
                messagebox.showwarning("Image Not Found", "Mapping image not found automatically.\nPlease select it manually.")
                selected = filedialog.askopenfilename(title="Select Image", filetypes=[("Image", "*.jpg *.png *.jpeg")])
                img_path = selected or None
            else:
                img_path = str(candidate)

            if img_path:
                calibration = self.calibrate_image(img_path)
            if calibration is None:
                options.generate_overlay = False

        output_dir = Path(map_file).parent / (Path(map_file).stem + "_Results")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(map_file, "r", encoding="utf-8") as file:
            lines = file.readlines()

        waves = [w for w in lines[0].strip().split("\t") if w]

        prog = tk.Toplevel(self.root)
        self.center_window(prog, 400, 100)
        pb = ttk.Progressbar(prog, length=300, maximum=len(lines) - 1)
        pb.pack(pady=20)
        img_data = mpimg.imread(img_path) if img_path and options.generate_overlay else None

        for i in range(1, len(lines)):
            parts = lines[i].strip().split("\t")
            y_val, x_val = float(parts[0]), float(parts[1])
            df = pd.DataFrame({"W": waves, "I": parts[2:]}).astype(float)
            plot_y = self._normalize(df["I"]) if options.normalize else df["I"]
            prefix = f"Point_{i}_X{x_val}_Y{y_val}"

            if options.split_txt:
                df.to_csv(output_dir / f"{prefix}.txt", sep="\t", index=False, header=False)
                if options.generate_individual_plot:
                    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
                    self.apply_plot_style(ax, df["W"], plot_y, options.normalize, options.peak_labeling, options.graphene_calc)
                    fig.savefig(output_dir / f"{prefix}_Plot.png", bbox_inches="tight")
                    plt.close(fig)

            if options.generate_overlay and calibration:
                s_x, o_x, s_y, o_y = calibration
                px, py = x_val * s_x + o_x, y_val * s_y + o_y

                fig_ov, (ax_i, ax_s) = plt.subplots(2, 1, figsize=(8, 10), dpi=150)
                ax_i.imshow(img_data)
                ax_i.plot(px, py, "bo", markersize=1, markeredgecolor="white", markeredgewidth=0.1, zorder=10)
                ax_i.axhline(y=py, color="white", linestyle="--", linewidth=0.5, alpha=0.7)
                ax_i.axvline(x=px, color="white", linestyle="--", linewidth=0.5, alpha=0.7)
                ax_i.axis("off")
                self.apply_plot_style(ax_s, df["W"], plot_y, options.normalize, options.peak_labeling, options.graphene_calc)
                fig_ov.savefig(output_dir / f"{prefix}_Overlay.png", bbox_inches="tight")
                plt.close(fig_ov)

            pb["value"] = i
            prog.update()

        prog.destroy()
        messagebox.showinfo("Success", "Mapping process completed.")

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    RamanProcessorApp().run()
