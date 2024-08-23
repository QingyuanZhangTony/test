# MyProject

## Project Overview

This project aims to provide a tool to utilize locally deployed Raspberry Shake devices to detect and report on catalogued earthquakes.

This program is designed and implemented for my MDS dissertation project.

It has two parts:
1. Daily Catalog Monitoring And Earthquake Detection.
2. Real-time Catalog Monitoring And Earthquake Detection (currently under construction and not included in this GUI).

## Table of Contents

- [Project Overview](#project-overview)
- [Installation Guide](#installation-guide)
- [Usage Instructions](#usage-instructions)
- [Configuration](#configuration)
- [Version Notes](#version-notes)
- [Known Issues](#known-issues)
- [Planned](#planned)

## Installation Guide

### Prerequisites

- [Anaconda or Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

### Steps

1. **Download the Project**

    Download the `MDS_Testing.zip` file and extract it to your desired location, for example `D:\MDS_Testing`.

2. **Navigate to the Project Directory**

    Open a terminal (Command Prompt or PowerShell) and navigate to the project directory:

    - If you are currently on a different drive (e.g., `C:` drive) and want to switch to `D:` drive, use the `/d` parameter with the `cd` command:

      ```bash
      cd /d D:\MDS_Testing
      ```

    - If you are already on the same drive as the project directory, you can use the `cd` command without the `/d` parameter:

      ```bash
      cd D:\MDS_Testing
      ```

3. **Create and Activate the Conda Environment**

    Create the Conda environment using the `environment.yml` file and activate it:

    ```bash
    conda env create -f environment.yml
    conda activate MDS_testing_env
    ```

## Usage Instructions

1. **Run the Main Script**

    After activating the Conda environment, you can run the main script using Streamlit:

    ```bash
    python -m streamlit run Home.py
    ```

    Alternatively, you can manually run the GUI by executing `startGUI.py`:

    ```bash
    python startGUI.py
    ```

    Both methods will open a browser window running the main GUI interface.

## Configuration

All settings are stored in the `default_config.yaml` configuration file. Every time the program runs, it automatically reads the parameter settings from this file. Before running the program for the first time, you need to replace the station information in this file with your own.

The current "Save setting" function in the GUI will not modify this configuration file; it only affects the settings in the session state.

The "Reset Settings" button will revert any modified parameters back to the values read from the configuration file.

The functionality to save and load custom configuration files will be added in the next version.

## Version Notes

- **v1.0**: Initial release with basic functionalities.

## Known Issues

- When the codes finish execution after a button click, the page might return to the top, and you have to manually scroll down to where you were.
- If you see errors about "session.states", go to "Settings" page and go back, and the error should go away. It will be fixed in the next version.

## Planned

- Allow users to save and load custom configuration files, and have changes made on the page be saved to the file.
