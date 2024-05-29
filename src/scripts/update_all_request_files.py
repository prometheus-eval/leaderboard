import json
import os
import time

from huggingface_hub import snapshot_download

from src.envs import API, DYNAMIC_INFO_FILE_PATH, DYNAMIC_INFO_PATH, DYNAMIC_INFO_REPO, EVAL_REQUESTS_PATH, H4_TOKEN
from src.submission.check_validity import check_model_card, get_model_tags, is_model_on_hub


def update_one_model(model_id, data, models_on_the_hub):
    # Model no longer on the hub at all
    if model_id not in models_on_the_hub:
        data["still_on_hub"] = False
        data["likes"] = 0
        data["downloads"] = 0
        data["created_at"] = ""
        data["tags"] = []
        return data

    # Grabbing model parameters
    model_cfg = models_on_the_hub[model_id]
    data["likes"] = model_cfg.likes
    data["downloads"] = model_cfg.downloads
    data["created_at"] = str(model_cfg.created_at)
    data["license"] = model_cfg.card_data.license if model_cfg.card_data is not None else ""

    # Grabbing model details
    model_name = model_id
    if model_cfg.card_data is not None and model_cfg.card_data.base_model is not None:
        if isinstance(model_cfg.card_data.base_model, str):
            model_name = model_cfg.card_data.base_model  # for adapters, we look at the parent model
    still_on_hub, _, _ = is_model_on_hub(
        model_name=model_name,
        revision=data.get("revision"),
        trust_remote_code=True,
        test_tokenizer=False,
        token=H4_TOKEN,
    )
    # If the model doesn't have a model card or a license, we consider it's deleted
    if still_on_hub:
        try:
            status, _, model_card = check_model_card(model_id)
            if status is False:
                still_on_hub = False
        except Exception:
            model_card = None
            still_on_hub = False
    data["still_on_hub"] = still_on_hub

    tags = get_model_tags(model_card, model_id) if still_on_hub else []

    data["tags"] = tags
    return data


def update_models(file_path, models_on_the_hub):
    """
    Search through all JSON files in the specified root folder and its subfolders,
    and update the likes key in JSON dict from value of input dict
    """
    seen_models = []
    with open(file_path, "r") as f:
        model_infos = json.load(f)
        for model_id in model_infos.keys():
            seen_models.append(model_id)
            model_infos[model_id] = update_one_model(
                model_id=model_id, data=model_infos[model_id], models_on_the_hub=models_on_the_hub
            )

    # If new requests files have been created since we started all this
    # we grab them
    all_models = []
    try:
        for ix, (root, _, files) in enumerate(os.walk(EVAL_REQUESTS_PATH)):
            if ix == 0:
                continue
            for file in files:
                if "eval_request" in file:
                    path = root.split("/")[-1] + "/" + file.split("_eval_request")[0]
                    all_models.append(path)
    except Exception as e:
        print(e)
        pass

    for model_id in all_models:
        if model_id not in seen_models:
            model_infos[model_id] = update_one_model(model_id=model_id, data={}, models_on_the_hub=models_on_the_hub)

    with open(file_path, "w") as f:
        json.dump(model_infos, f, indent=2)


def update_dynamic_files():
    """This will only update metadata for models already linked in the repo, not add missing ones."""
    snapshot_download(
        repo_id=DYNAMIC_INFO_REPO, local_dir=DYNAMIC_INFO_PATH, repo_type="dataset", tqdm_class=None, etag_timeout=30
    )

    print("UPDATE_DYNAMIC: Loaded snapshot")
    # Get models
    start = time.time()

    models = list(
        API.list_models(
            # filter=ModelFilter(task="text-generation"),
            full=False,
            cardData=True,
            fetch_config=True,
        )
    )
    id_to_model = {model.id: model for model in models}

    print(f"UPDATE_DYNAMIC: Downloaded list of models in {time.time() - start:.2f} seconds")

    start = time.time()

    update_models(DYNAMIC_INFO_FILE_PATH, id_to_model)

    print(f"UPDATE_DYNAMIC: updated in {time.time() - start:.2f} seconds")

    API.upload_file(
        path_or_fileobj=DYNAMIC_INFO_FILE_PATH,
        path_in_repo=DYNAMIC_INFO_FILE_PATH.split("/")[-1],
        repo_id=DYNAMIC_INFO_REPO,
        repo_type="dataset",
        commit_message="Daily request file update.",
    )
    print("UPDATE_DYNAMIC: pushed to hub")
