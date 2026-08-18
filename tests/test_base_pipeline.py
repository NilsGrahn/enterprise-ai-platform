"""The BasePipeline contract — the tests that protect the platform claim."""


import pytest
from ml_service.base_pipeline import BasePipeline, SchemaError, TrainingResult
from ml_service.pipelines import PIPELINE_REGISTRY, get_pipeline


class TestAbstractness:

    def test_base_pipeline_cannot_be_instantiated(self):
        with pytest.raises(TypeError) as exc:
            BasePipeline()
        message = str(exc.value)
        assert 'abstract' in message.lower()

    def test_incomplete_subclass_cannot_be_instantiated(self):
        class Incomplete(BasePipeline):
            name = 'incomplete'
            target_column = 'y'
            source_table = 'nowhere'
            # deliberately implements nothing

        with pytest.raises(TypeError):
            Incomplete()


class TestRegistryContract:
    """If any of these fail, the platform claim from ADR-0004 is broken."""

    def test_registry_is_not_empty(self):
        assert len(PIPELINE_REGISTRY) > 0

    @pytest.mark.parametrize('name', list(PIPELINE_REGISTRY))
    def test_every_entry_subclasses_base_pipeline(self, name):
        assert issubclass(PIPELINE_REGISTRY[name], BasePipeline)

    @pytest.mark.parametrize('name', list(PIPELINE_REGISTRY))
    def test_every_entry_defines_its_identity(self, name):
        pipeline = get_pipeline(name)
        assert isinstance(pipeline.name, str) and pipeline.name
        assert isinstance(pipeline.target_column, str) and pipeline.target_column
        assert isinstance(pipeline.source_table, str) and pipeline.source_table

    @pytest.mark.parametrize('name', list(PIPELINE_REGISTRY))
    def test_registry_key_matches_pipeline_name(self, name):
        assert get_pipeline(name).name == name

    @pytest.mark.parametrize('name', list(PIPELINE_REGISTRY))
    def test_required_columns_is_a_non_empty_list(self, name):
        columns = get_pipeline(name).required_columns
        assert isinstance(columns, list) and len(columns) > 0
        assert all(isinstance(c, str) for c in columns)

    @pytest.mark.parametrize('name', list(PIPELINE_REGISTRY))
    def test_all_four_methods_are_implemented(self, name):
        pipeline = get_pipeline(name)
        for method in ('clean', 'feature_engineering', 'train', 'predict'):
            assert callable(getattr(pipeline, method))
            assert not getattr(getattr(pipeline, method), '__isabstractmethod__', False)

    def test_unknown_pipeline_raises(self):
        from ml_service.pipelines import UnknownPipelineError
        with pytest.raises(UnknownPipelineError):
            get_pipeline('does_not_exist')


class TestTemplateMethodIsNotOverridden:
    """run_training defines the training protocol. Subclasses must not change it."""

    @pytest.mark.parametrize('name', list(PIPELINE_REGISTRY))
    def test_run_training_comes_from_the_base_class(self, name):
        cls = PIPELINE_REGISTRY[name]
        assert 'run_training' not in cls.__dict__, (
            f"{cls.__name__} overrides run_training. The training protocol is "
            f"fixed by BasePipeline — see ADR-0004."
        )

    @pytest.mark.parametrize('name', list(PIPELINE_REGISTRY))
    def test_load_data_comes_from_the_base_class(self, name):
        cls = PIPELINE_REGISTRY[name]
        assert 'load_data' not in cls.__dict__


class TestSchemaValidation:

    def test_missing_columns_raise_schema_error(self, sample_gold_df):
        pipeline = get_pipeline('credit')
        df = sample_gold_df.drop(columns=['monthly_income'])
        with pytest.raises(SchemaError) as exc:
            pipeline.validate_schema(df)
        assert 'monthly_income' in str(exc.value)

    def test_complete_frame_passes(self, sample_gold_df):
        get_pipeline('credit').validate_schema(sample_gold_df)


class TestFeatureFrameHelper:
    """build_feature_frame was added in Phase 7 for prediction logging."""

    def test_returns_the_frozen_column_order(self, trained_pipeline, single_applicant_df):
        X = trained_pipeline.build_feature_frame(single_applicant_df)
        assert list(X.columns) == trained_pipeline.preprocessing['feature_names']

    def test_does_not_mutate_stored_preprocessing(self, trained_pipeline,
                                                  single_applicant_df):
        before = dict(trained_pipeline.preprocessing)
        trained_pipeline.build_feature_frame(single_applicant_df)
        assert trained_pipeline.preprocessing == before


class TestTrainingResult:

    def test_has_the_expected_fields(self):
        result = TrainingResult(model=None, metrics={}, feature_names=[],
                                preprocessing={}, training_rows=0)
        for field in ('model', 'metrics', 'feature_names', 'preprocessing',
                      'training_rows'):
            assert hasattr(result, field)