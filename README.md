# EDI to ModEM with MTpy-v2

A Python/Tkinter workflow for converting magnetotelluric (MT) EDI files into ModEM input files using MTpy-v2.

The workflow provides a graphical interface for:

- Reading a folder of EDI files
- Building a 3-D structured ModEM mesh with user-defined core cells and padding
- Selecting a common target-frequency grid
- Centering MT stations on model cells
- Applying topography from automatic SRTM/ETOPO data or a local GeoTIFF
- Using a user-defined number and constant thickness of air layers
- Generating an initial 3-D resistivity model, ModEM data file, and covariance file
- Creating one mesh-exact vertical resistivity section inside the core model region

## Main files

```text
edi_to_modem_gui_mtpy_v2.py   # Tkinter GUI
edi_to_modem_mtpy_v2.py       # Main EDI -> ModEM workflow
auto_topography.py             # SRTM / ETOPO / GeoTIFF topography handling
environment.yml                # Conda environment
requirements.txt               # Python package requirements
README.md
```

The three Python files above must remain in the same directory.

## Requirements

The recommended environment uses:

- Python 3.11
- MTpy-v2 2.1.4
- Tk/Tkinter
- NumPy
- SciPy
- Pandas
- Matplotlib
- Rasterio
- PyProj
- Requests

MTpy-v2 officially supports Python 3.10 and newer, and the stable 2.1.4 release can be installed with either pip or conda-forge. This project pins MTpy-v2 to 2.1.4 because the workflow was developed and tested against that API.

## 1. Install Git

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install git
```

Check:

```bash
git --version
```

## 2. Install Miniforge / Conda

If Conda is already installed, skip this step.

Miniforge is recommended because it works well with conda-forge packages and geospatial dependencies such as Rasterio.

After installing Conda, restart the terminal and check:

```bash
conda --version
```

## 3. Create the MTpy-v2 environment

### Option A: use the supplied environment file

From the repository directory:

```bash
conda env create -f environment.yml
conda activate mtpy-v2-modem
```

### Option B: create the environment manually

```bash
conda create -n mtpy-v2-modem -c conda-forge python=3.11 mtpy-v2=2.1.4 numpy pandas matplotlib scipy rasterio pyproj requests tk
conda activate mtpy-v2-modem
```

The `tk` package provides Tcl/Tk for the Tkinter GUI when the Conda environment is used.

## 4. Check MTpy-v2 and Tkinter

Check MTpy-v2:

```bash
python -c "import importlib.metadata as m; print('MTpy-v2:', m.version('mtpy-v2'))"
```

Expected:

```text
MTpy-v2: 2.1.4
```

Check Tkinter:

```bash
python -c "import tkinter; print('Tkinter: OK')"
```

You can also test the Tk GUI directly with:

```bash
python -m tkinter
```

A small Tkinter test window should open.

### Ubuntu system Tkinter alternative

If Tkinter is missing from a system Python installation, Ubuntu provides the `python3-tk` package:

```bash
sudo apt install python3-tk
```

When using the Conda environment in this repository, prefer the Conda `tk` package shown above.

## 5. Clone this repository

Once the repository has been created on GitHub:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

Activate the environment:

```bash
conda activate mtpy-v2-modem
```

## 6. Run the GUI

Run:

```bash
python edi_to_modem_gui_mtpy_v2.py
```

The GUI will open and allow you to select:

### Input / output

- EDI folder
- Output folder
- WGS84 UTM EPSG code

### Frequency grid

- Number of target frequencies
- Minimum frequency
- Maximum frequency

The workflow creates a common logarithmic target-frequency grid and uses only the frequency range supported by the available MT data; it does not extrapolate beyond the data range.

### 3-D mesh

- Core cells in East and North directions
- Core cell size in East and North directions
- Number of padding cells on each side
- Horizontal padding growth factor
- Number of Earth layers
- First Earth-layer thickness
- Vertical Earth-layer growth factor

### Topography

Four topography modes are available:

1. **Automatic (SRTM → ETOPO)**
2. **SRTM 30 m**
3. **ETOPO 2022 15 arc-sec**
4. **Local GeoTIFF**

For automatic mode, the program determines the required geographic extent from the model and selects SRTM when available, with ETOPO 2022 as the fallback.

Internet access is required when a DEM is not already available in the local cache.

### Air layers

When topography is enabled, the user specifies both:

- Number of air layers
- Thickness of each air layer

For example:

```text
Number of air layers = 32
Air layer thickness  = 100 m
```

creates exactly 32 air cells, each 100 m thick. The workflow does not use a geometric growth factor for the air cells.

### Model section

Only one vertical resistivity section is generated. The user specifies the maximum section depth below sea level.

The plotted section uses the actual model-cell boundaries and is restricted to the main core mesh rather than the horizontal padding region. MT stations are displayed at their projected ground elevations.

## 7. Command-line mode

The main workflow can also be run without the GUI.

Interactive mode:

```bash
python edi_to_modem_mtpy_v2.py
```

Using a JSON configuration file:

```bash
python edi_to_modem_mtpy_v2.py --config path/to/config.json
```

## 8. Expected outputs

The program creates a ModEM-ready output directory containing, among other files:

```text
ModEM_Model_File.rho
ModEM_Data.dat
covariance.cov
stations.csv
selected_frequencies_Hz.csv
selected_frequencies_Hz.txt
run_config.json
run_summary.json
figures/
    vertical_section_mesh.png
```

Additional topography and station-diagnostic files may be created when topography is enabled.

## 9. Data handling

Do not commit project-specific EDI files, DEM files, large raster files, ModEM outputs, or local run directories to GitHub unless you intentionally want to publish them.

The repository `.gitignore` excludes common MT/ModEM data and generated output files.



## Scientific / software notes

The coordinate convention used by the workflow follows the ModEM/MTpy-v2 model convention documented in the source code:

- X = North
- Y = East
- Z = positive downward

The workflow is intended for preparation of ModEM input files, not for performing the ModEM inversion itself.

## MTpy-v2

This project uses MTpy-v2:

https://github.com/MTgeophysics/mtpy-v2

MTpy-v2 documentation:

https://mtpy-v2.readthedocs.io/

## License

edi-to-modem-mtpy-v2 is licensed under the MIT license

The license agreement is contained in the repository and should be kept together with the code.
