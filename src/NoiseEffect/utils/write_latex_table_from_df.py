import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Number / text formatting helpers
# ---------------------------------------------------------------------------

_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape(text: str) -> str:
    """Escape LaTeX special characters in a plain-text string."""
    out = []
    for ch in text:
        out.append(_LATEX_SPECIALS.get(ch, ch))
    return "".join(out)


def _fmt_number(x, decimals, sci_low, sci_high):
    """
    Format a single numeric scalar.
    - integer-typed values  -> plain integer
    - floats of normal size -> fixed decimals
    - very small / large    -> $m \\times 10^{e}$
    """
    if isinstance(x, bool):
        return str(x)

    # Integer types render as integers (no decimals, no sci notation).
    if isinstance(x, (int, np.integer)):
        return f"{int(x)}"

    xf = float(x)
    if np.isnan(xf):
        return ""
    if np.isinf(xf):
        return r"$\infty$" if xf > 0 else r"$-\infty$"
    if xf == 0:
        return "0"

    ax = abs(xf)
    if ax < sci_low or ax >= sci_high:
        mant, exp = f"{xf:.{decimals}e}".split("e")
        return f"${mant} \\times 10^{{{int(exp)}}}$"
    return f"{xf:.{decimals}f}"


def format_dataframe_for_latex(
    df: pd.DataFrame,
    decimals: int = 3,
    sci_low: float = 1e-3,
    sci_high: float = 1e5,
    escape_text: bool = True,
) -> pd.DataFrame:
    """
    Return a copy of df with every value turned into a display string.

    Formatting is inferred from the data, so this works on any DataFrame:
      - integer columns          -> integers
      - float columns            -> `decimals` places, power-of-10 for tiny/huge
      - object / mixed columns   -> decided per cell by the value's own type
        (this is what makes it survive a .T transpose, where columns become
        object dtype holding a mix of ints, floats and strings)
    Non-numeric text is escaped for LaTeX when escape_text=True.
    """
    out = df.copy()

    def fmt_cell(v):
        if pd.isna(v):
            return ""
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(
            v, bool
        ):
            return _fmt_number(v, decimals, sci_low, sci_high)
        text = str(v)
        return _latex_escape(text) if escape_text else text

    for col in out.columns:
        s = out[col]
        if pd.api.types.is_bool_dtype(s):
            out[col] = s.map(lambda v: "" if pd.isna(v) else str(v))
        elif pd.api.types.is_integer_dtype(s):
            out[col] = s.map(lambda v: "" if pd.isna(v) else f"{int(v)}")
        elif pd.api.types.is_float_dtype(s):
            out[col] = s.map(lambda v: _fmt_number(v, decimals, sci_low, sci_high))
        else:  # object / mixed / string
            out[col] = s.map(fmt_cell)

    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def df_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    column_format: str | None = None,
    index: bool = False,
    decimals: int = 3,
    sci_low: float = 1e-3,
    sci_high: float = 1e5,
    resize: bool = True,
    force_placement: bool = True,
    clean_headers: bool = True,
    escape_text: bool = True,
    format_numbers: bool = True,
    **kwargs,
):
    """
    Convert any DataFrame to a LaTeX table with standard styling.

    Parameters
    ----------
    index : bool
        If True, the DataFrame index is shown as a first (left-aligned) column.
        Use this when your row labels live in the index (e.g. a transposed
        properties table).
    column_format : str | None
        LaTeX column spec. If None, it's built automatically as "l" + "c"*(k-1)
        for the final number of columns.
    decimals, sci_low, sci_high :
        Float precision and the magnitude thresholds below/above which a value
        is rendered in power-of-10 form.
    clean_headers : bool
        Replace underscores with spaces in column headers (no title-casing, so
        names like "PPI (ER)" are preserved).
    escape_text : bool
        Escape LaTeX special characters in headers, index labels and any
        non-numeric cell text. Numeric cells (which may contain generated math
        like $1.2 \\times 10^{-3}$) are never escaped.
    """
    df_copy = df.copy()

    # Promote the index into a real first column if requested, so it gets the
    # same cleaning/escaping and is counted in column_format.
    if index:
        header = df_copy.index.name if df_copy.index.name is not None else ""
        idx_vals = [str(v) for v in df_copy.index]
        if clean_headers:
            idx_vals = [v.replace("_", " ") for v in idx_vals]
        df_copy = df_copy.reset_index(drop=True)
        df_copy.insert(0, header, idx_vals)

    if format_numbers:
        # Don't re-format the promoted label column as a number; it's text.
        label_col = df_copy.columns[0] if index else None
        target = df_copy.drop(columns=[label_col]) if index else df_copy
        formatted = format_dataframe_for_latex(
            target,
            decimals=decimals,
            sci_low=sci_low,
            sci_high=sci_high,
            escape_text=escape_text,
        )
        if index:
            labels = df_copy[[label_col]].copy()
            if escape_text:
                labels[label_col] = labels[label_col].map(
                    lambda v: _latex_escape(str(v))
                )
            df_copy = pd.concat([labels, formatted], axis=1)
        else:
            df_copy = formatted

    # Headers: clean underscores then escape.
    new_cols = []
    for c in df_copy.columns:
        c = str(c)
        if clean_headers:
            c = c.replace("_", " ")
        if escape_text:
            c = _latex_escape(c)
        new_cols.append(c)
    df_copy.columns = new_cols

    # Auto column format.
    if column_format is None:
        k = df_copy.shape[1]
        column_format = "l" + "c" * (k - 1) if k > 1 else "l"

    latex_str = df_copy.to_latex(
        index=False,
        caption=caption,
        label=label,
        column_format=column_format,
        escape=False,  # we escape ourselves so generated math survives
        **kwargs,
    )

    if resize:
        latex_str = latex_str.replace(
            "\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}"
        )
        latex_str = latex_str.replace("\\end{tabular}", "\\end{tabular}%\n}")

    if force_placement:
        latex_str = latex_str.replace("\\begin{table}", "\\begin{table}[H]")

    return latex_str


def save_latex_tables(file_path: str, tables: "str | list[str]"):
    """Write one or more LaTeX table strings to a .tex file."""
    if isinstance(tables, str):
        tables = [tables]
    with open(file_path, "w") as f:
        f.write("\n\n".join(tables))
