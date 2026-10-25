"""Small sanity tests for the ML workflow."""

from sklearn.model_selection import train_test_split

from src.train import RANDOM_STATE, TEST_SIZE, build_models, load_dataset


def test_dataset_has_expected_shape_and_two_classes():
    X, y, class_names = load_dataset()
    assert X.shape == (569, 30)
    assert y.nunique() == 2
    assert len(class_names) == 2


def test_split_is_stratified_and_reproducible():
    X, y, _ = load_dataset()
    split1 = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    split2 = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    assert split1[0].equals(split2[0])
    assert len(split1[1]) == 114


def test_build_models_returns_two_models():
    models = build_models()
    assert set(models) == {"Logistic Regression", "Random Forest"}
