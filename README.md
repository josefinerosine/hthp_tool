# hthp-mvr-project

A Streamlit-based modelling and pre-selection tool for industrial steam and
hot-water supply. It calculates and compares different system architectures for
producing hot steam or water — **HTHP**, **HTHP + MVR**, **MVR alone**, or
**electric generation** — and evaluates different natural refrigerants and
compressor types to optimise efficiency and performance for industrial heat and
steam applications.

The tool is organised as a five-page Streamlit application:

1. **Upload** – load process parameters (questionnaire file or manual entry)
2. **Pinch analysis** – composite curves and minimum utility demand
3. **Case generation** – configure the system cases to be simulated
4. **Calculation** – run the TESPy-based thermodynamic simulation
5. **Results** – per-case detail views and a comparative overview

---

## Installation

**1. Clone the repository (including submodules)**

The `heatpumps` library is included as a Git submodule, so clone with
`--recurse-submodules`:

```bash
git clone --recurse-submodules <repository-url>
```

If you already cloned without submodules, fetch them afterwards:

```bash
git submodule update --init --recursive
```

**2. Install Streamlit**

```bash
pip install streamlit
```

**3. Install the remaining dependencies**

The tool additionally requires the following packages:

```bash
pip install pandas numpy matplotlib plotly CoolProp tespy fluprodia openpyxl
```

> `openpyxl` is needed to read the `.xlsx` questionnaire file. The `heatpumps`
> submodule provides the vapour-compression cycle models used internally.

---

## Running the tool

Change into the project folder and start the application with:

```bash
streamlit run hthp_project/streamlitTool.py
```

Streamlit will open the tool in your web browser (by default at
`http://localhost:8501`).

---

## Providing process parameters

There are two ways to supply the system specifications on the **Upload** page:

- **Questionnaire file** – fill in the provided `hthp_questionnaire.xlsx` file
  and upload it in the tool.
- **Manual entry** – alternatively, use the corresponding page in the tool to
  enter the parameters manually. Entered parameters can be saved (and re-loaded)
  so you do not have to re-type them on the next run.

Both options feed the same downstream pinch analysis, case generation and
simulation steps.
