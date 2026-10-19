from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from aura.config import AudioConfig
from experiments.cost_confirmation import retention_budget,make_speaker_folds,paired_order
from experiments import raw_pipeline
from scripts import study_cost_confirmation as study
from scripts import report_cost_confirmation as report


def test_fraction_transfer_and_balanced_rounding():
    assert retention_budget(17500,10) == 6400
    assert retention_budget(2500,10) == 920
    with pytest.raises(ValueError):
        retention_budget(0,10)


def test_loso_removes_cross_speaker_content_and_covers_each_speaker_once():
    records=[dict(id=f"{s}{label}{i}",speaker_id=s,label=str(label),audio_hash=f"{s}{label}{i}")
        for s in "abc" for label in range(2) for i in range(3)]
    records[6]["audio_hash"]=records[0]["audio_hash"]
    folds=make_speaker_folds(records)
    byid={r["id"]:r for r in records}
    assert {f["speaker"] for f in folds} == set("abc")
    for fold in folds:
        assert not {byid[i]["audio_hash"] for i in fold["train_ids"]} & {byid[i]["audio_hash"] for i in fold["test_ids"]}
        assert all(byid[i]["speaker_id"] != fold["speaker"] for i in fold["train_ids"])
    assert sum(len(f["test_ids"]) for f in folds) == len(records)
    records[6]["label"]="bad"
    with pytest.raises(ValueError,match="conflicting"):
        make_speaker_folds(records)


def test_repeat_order_counterbalances_first_method():
    orders=[paired_order(fold,repeat) for fold in range(6) for repeat in range(3)]
    assert sum(order[0] == "full_pool" for order in orders) == 9
    assert sum(order[0] == "random" for order in orders) == 9


def test_raw_encoding_order_is_restored_across_chunks(monkeypatch):
    monkeypatch.setattr(raw_pipeline,"process_audio",lambda path,**kw:SimpleNamespace(waveform=np.ones(int(path))))
    class Fake:
        batch_size=2
        info=SimpleNamespace(embedding_dim=1)
        def encode_batch(self,waves,sr):
            return np.array([[len(w)] for w in waves],dtype=np.float32)
    meta=pd.DataFrame(dict(id=list("abcd"),audio_path=["9","3","7","4"]))
    result=raw_pipeline.encode_recordings(meta,Fake(),AudioConfig(),chunk_size=3)
    np.testing.assert_array_equal(result[:,0],[9,3,7,4])


def test_all_primary_models_and_timing_repeats_are_predeclared():
    folds=[dict(speaker=str(i),train_ids=list(range(2500)),random_budget=920) for i in range(6)]
    jobs=study.jobs_for(folds)
    assert len(jobs) == 60
    assert sum(j["primary"] for j in jobs) == 36
    assert sum(j["timed_pair"] for j in jobs) == 36
    assert all(j["timed_pair"] for j in jobs[:36])
    assert len({(j["speaker"],j["method"],j["seed"],j["repeat"]) for j in jobs}) == 60


def test_trial_checkpoint_reuses_and_rejects_changed_artifacts(tmp_path,monkeypatch):
    meta=pd.DataFrame(dict(id=[str(i) for i in range(20)],label=["0","1"]*10,split="train"))
    monkeypatch.setattr(study,"sync",lambda:None)
    monkeypatch.setattr(study,"encode_recordings",lambda selected,*a:np.tile(np.array([[1,0]],dtype=np.float32),(len(selected),1)))
    monkeypatch.setattr(study,"fit_probe",lambda *a:({"dummy":"model"},.001))
    job=dict(speaker="speaker",method="random",budget=10,seed=0,repeat=0,C=1.0,timed_pair=True,primary=True)
    config=study.AuraConfig()
    folder=study.train_trial(job,meta,None,config,tmp_path,"protocol","session")
    assert study.train_trial(job,meta,None,config,tmp_path,"protocol","session") == folder
    (folder/"selection.json").write_text('["changed"]')
    with pytest.raises(ValueError,match="Changed"):
        study.train_trial(job,meta,None,config,tmp_path,"protocol","session")


def test_primary_gap_averages_seeds_before_comparing_recordings():
    rows=[]
    for speaker in "ab":
        rows.append(dict(id=speaker,speaker_id=speaker,method="full_pool",seed=0,primary=True,label="0",prediction="0"))
        rows.extend(dict(id=speaker,speaker_id=speaker,method="random",seed=seed,primary=True,label="0",prediction="0" if seed < 4 else "1") for seed in range(5))
    frame=pd.DataFrame(rows)
    result,by_speaker=report.primary_gap(frame)
    assert result["recordings"] == 2
    assert result["gap_pp"] == pytest.approx(20)
    assert result["random_accuracy"] == pytest.approx(.8)
    assert by_speaker.recordings.tolist() == [1,1]
    with pytest.raises(ValueError,match="five subset seeds"):
        report.primary_gap(frame.drop(frame.index[-1]))
    with pytest.raises(ValueError,match="duplicated"):
        report.primary_gap(pd.concat([frame,frame.iloc[[0]]]))


def test_timing_pairs_requires_complete_same_session_pairs():
    frame=pd.DataFrame([dict(speaker="a",repeat=0,method=method,session="one",timed_pair=True,end_to_end_seconds=seconds)
        for method,seconds in [("full_pool",10),("random",4)]])
    pairs=report.timing_pairs(frame)
    assert pairs.ratio.iloc[0] == pytest.approx(.4)
    assert pairs.saving_percent.iloc[0] == pytest.approx(60)
    with pytest.raises(ValueError,match="incomplete"):
        report.timing_pairs(frame.iloc[:1])
    frame.loc[1,"session"]="two"
    with pytest.raises(ValueError,match="different runtime sessions"):
        report.timing_pairs(frame)
