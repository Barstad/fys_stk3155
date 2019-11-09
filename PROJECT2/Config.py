import pathlib

class Config:
    DATA_SIZE = 100
    POLY = 5

    # USE_DATA_SIZE = 10000

    ############## FILL IN #############
    EXPERIMENT = "TERRAIN"
    ####################################

    assert EXPERIMENT in ['FRANKE', 'NOISY_FRANKE', 'TERRAIN']

    DATA_BASE_PATH = pathlib.Path('../PROJECT1/data')
    RAW_TERRAIN_DATA_FILE = 'SRTM_data_Norway_1.tif'

    # Data files
    TERRAIN_DATA = DATA_BASE_PATH.joinpath("terrain_data.csv")
    FRANKE_DATA_NOISY = DATA_BASE_PATH.joinpath("franke_noisy_data.csv")
    FRANKE_DATA = DATA_BASE_PATH.joinpath("franke_data.csv")

    OUTPUT_FOLDER = pathlib.Path("./results")
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    FRANKE_RESULTS = OUTPUT_FOLDER.joinpath("./franke")
    FRANKE_RESULTS.mkdir(exist_ok = True)
    NOISY_FRANKE_RESULTS = OUTPUT_FOLDER.joinpath("./noisy_franke")
    NOISY_FRANKE_RESULTS.mkdir(exist_ok = True)
    TERRAIN_RESULTS = OUTPUT_FOLDER.joinpath("./terrain")
    TERRAIN_RESULTS.mkdir(exist_ok = True)

    if EXPERIMENT == 'FRANKE':
        ACTIVE_DATA = FRANKE_DATA
        RESULT_FOLDER = FRANKE_RESULTS

    elif EXPERIMENT == "NOISY_FRANKE":
        ACTIVE_DATA = FRANKE_DATA_NOISY
        RESULT_FOLDER = NOISY_FRANKE_RESULTS

    elif EXPERIMENT == 'TERRAIN':
        ACTIVE_DATA = TERRAIN_DATA
        RESULT_FOLDER = TERRAIN_RESULTS
    else:
        raise ValueError


cfg = Config()