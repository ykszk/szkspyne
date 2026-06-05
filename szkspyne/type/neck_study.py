from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, RootModel


class SortedFiles(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flexion: Optional[str] = None
    neutral: Optional[str] = None
    extension: Optional[str] = None


class UnsortedFiles(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unsorted: list[str]


class Studies(RootModel[dict[str, Union[SortedFiles, UnsortedFiles]]]):
    pass


if __name__ == "__main__":
    import json
    import sys

    from logzero import logger
    
    args = sys.argv
    if len(args) < 3:
        print("Usage: python neck_study.py <neck_studies.json> <ndjson>")
        sys.exit(1)
    json_path = args[1]
    with open(json_path, 'r') as f:
        data = json.load(f)

    studies = Studies.model_validate(data)
    logger.info(f"Loaded {len(studies.model_dump())} studies from {json_path}")

    ndjson_path = args[2]
    all_data = {}
    with open(ndjson_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            filename = data['filename']
            all_data[filename] = data
    logger.info(f"Loaded {len(all_data)} entries from {ndjson_path}")    

    dump_count = 0
    for study_name, study_files in studies.model_dump().items():
        logger.debug(f"Study: {study_name}")
        # if isinstance(study_files, dict):
        if 'unsorted' not in study_files:
            for pose in ['neutral', 'flexion', 'extension']:
                file = study_files[pose]
                if file is None:
                    continue
                data = all_data.get(file, None)
                if data is None:
                    # logger.warning(f"  File {file} not found in {ndjson_path}")
                    continue
                print(json.dumps(data, ensure_ascii=False))
                dump_count += 1
        else:
            logger.debug("  Unsorted files:")
            for file in study_files['unsorted']:
                data = all_data.get(file, None)
                if data is None:
                    # logger.warning(f"  File {file} not found in {ndjson_path}")
                    continue
                print(json.dumps(data, ensure_ascii=False))
                dump_count += 1
        print()  # Add a newline between studies
    logger.info(f"Dumped {dump_count} entries from {ndjson_path}")