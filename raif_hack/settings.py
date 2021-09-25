from pathlib import Path

ROOT_DIR = Path(__file__).parents[0].parents[0]
DATA_DIR = ROOT_DIR / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = DATA_DIR / "submission.csv"


TARGET = "per_square_meter_price"
# признаки (или набор признаков), для которых применяем smoothed target encoding
CATEGORICAL_STE_FEATURES = ["region", "city", "realty_type"]
CATEGORICAL_LE_FEATURES = ["region", "city", "realty_type", "price_type"]

# признаки, для которых применяем one hot encoding
CATEGORICAL_OHE_FEATURES = ["region", "realty_type", "price_type"]  # type: ignore

# численные признаки
NUM_FEATURES = [
    "lat",
    "lng",
    "osm_amenity_points_in_0.001",
    "osm_amenity_points_in_0.005",
    "osm_amenity_points_in_0.0075",
    "osm_amenity_points_in_0.01",
    "osm_building_points_in_0.001",
    "osm_building_points_in_0.005",
    "osm_building_points_in_0.0075",
    "osm_building_points_in_0.01",
    "osm_catering_points_in_0.001",
    "osm_catering_points_in_0.005",
    "osm_catering_points_in_0.0075",
    "osm_catering_points_in_0.01",
    "osm_city_closest_dist",
    "osm_city_nearest_population",
    "osm_crossing_closest_dist",
    "osm_crossing_points_in_0.001",
    "osm_crossing_points_in_0.005",
    "osm_crossing_points_in_0.0075",
    "osm_crossing_points_in_0.01",
    "osm_culture_points_in_0.001",
    "osm_culture_points_in_0.005",
    "osm_culture_points_in_0.0075",
    "osm_culture_points_in_0.01",
    "osm_finance_points_in_0.001",
    "osm_finance_points_in_0.005",
    "osm_finance_points_in_0.0075",
    "osm_finance_points_in_0.01",
    "osm_healthcare_points_in_0.005",
    "osm_healthcare_points_in_0.0075",
    "osm_healthcare_points_in_0.01",
    "osm_historic_points_in_0.005",
    "osm_historic_points_in_0.0075",
    "osm_historic_points_in_0.01",
    "osm_hotels_points_in_0.005",
    "osm_hotels_points_in_0.0075",
    "osm_hotels_points_in_0.01",
    "osm_leisure_points_in_0.005",
    "osm_leisure_points_in_0.0075",
    "osm_leisure_points_in_0.01",
    "osm_offices_points_in_0.001",
    "osm_offices_points_in_0.005",
    "osm_offices_points_in_0.0075",
    "osm_offices_points_in_0.01",
    "osm_shops_points_in_0.001",
    "osm_shops_points_in_0.005",
    "osm_shops_points_in_0.0075",
    "osm_shops_points_in_0.01",
    "osm_subway_closest_dist",
    "osm_train_stop_closest_dist",
    "osm_train_stop_points_in_0.005",
    "osm_train_stop_points_in_0.0075",
    "osm_train_stop_points_in_0.01",
    "osm_transport_stop_closest_dist",
    "osm_transport_stop_points_in_0.005",
    "osm_transport_stop_points_in_0.0075",
    "osm_transport_stop_points_in_0.01",
    "reform_count_of_houses_1000",
    "reform_count_of_houses_500",
    "reform_house_population_1000",
    "reform_house_population_500",
    "reform_mean_floor_count_1000",
    "reform_mean_floor_count_500",
    "reform_mean_year_building_1000",
    "reform_mean_year_building_500",
    "total_square",
]

MODEL_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.01,
    reg_alpha=1,
    num_leaves=40,
    min_child_samples=5,
    importance_type="gain",
    n_jobs=1,
    random_state=563,
)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"},
    },
    "handlers": {
        "file_handler": {
            "level": "INFO",
            "formatter": "default",
            "class": "logging.FileHandler",
            "filename": "train.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {"handlers": ["file_handler"], "level": "INFO", "propagate": False},
    },
}
