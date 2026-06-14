from argparse import Namespace

import pipeline


def test_format_duration():
    assert pipeline.format_duration(7) == "7s"
    assert pipeline.format_duration(125) == "2m 5s"
    assert pipeline.format_duration(3725) == "1h 2m 5s"


def test_should_skip_stage_flags():
    args = Namespace(skip_preprocess=True, skip_train=False, skip_eval=True)

    assert pipeline.should_skip_stage("preprocess", args)
    assert not pipeline.should_skip_stage("train", args)
    assert pipeline.should_skip_stage("evaluate", args)


def test_output_metadata_collects_stage_outputs(tmp_path):
    output_file = tmp_path / "artifact.txt"
    output_file.write_text("ok")

    metadata = pipeline.output_metadata(
        [
            {
                "name": "dummy",
                "outputs": [output_file],
            }
        ]
    )

    assert metadata[0]["exists"] is True
    assert metadata[0]["type"] == "file"
    assert metadata[0]["size_bytes"] == 2


def test_pipeline_packages_inference_artifact_after_training():
    stage_names = [stage["name"] for stage in pipeline.STAGES]

    assert stage_names.index("train") < stage_names.index("package_inference")
    assert stage_names.index("package_inference") < stage_names.index("evaluate")
