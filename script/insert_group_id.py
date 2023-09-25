import argparse
import csv
import json
import re
import sys
from pathlib import Path
from argparse import ArgumentParser
from abc import ABC, abstractmethod

import numpy as np


class DictMapper(ABC):

    def __init__(self, args):
        self.args = args

    @property
    @classmethod
    @abstractmethod
    def name(cls) -> str:
        pass

    @property
    @classmethod
    @abstractmethod
    def help(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def add_arguments(cls, parser: ArgumentParser):
        pass

    @abstractmethod
    def map(self, data: dict) -> dict:
        pass

    @abstractmethod
    def finalize(self):
        pass

def filepath2groupid(filepath: str) -> str:
    filepath = Path(filepath.replace('\\', '/')).name
    if filepath.startswith("OSC"):
        group_id = filepath.split("_")[0]
    elif filepath.startswith("CMCAISY"):
        m = re.search(r"\d+", filepath)
        if m is None:
            raise RuntimeError("No number found in: " + filepath)
        group_id = "CMCAISY" + m.group()
    elif filepath.startswith('DELAISY'):
        m = re.search(r'DE?L?(\d+)', filepath.upper())
        if m is None:
            group_id = re.split(r"-|=|\.|,| |_|yr", filepath)[0]
        else:
            group_id = 'DE' + m[1]
    else:
        group_id = re.split(r"-|=|\.|,| |_|yr", filepath)[0]
    return group_id

class InsertGroupId(DictMapper):

    def __init__(self, *args):
        super().__init__(*args)
        self.path2id = []

    @classmethod
    @property
    def name(cls) -> str:
        return 'insert_group_id'

    @classmethod
    @property
    def help(cls) -> str:
        return 'Extract and insert group id'

    @classmethod
    def add_arguments(cls, parser: ArgumentParser):
        parser.add_argument(
            "--src_key",
            help="Extract group id from this key. default: %(default)s",
            default="filepath")
        parser.add_argument(
            "--id_key",
            help="Insert group id into this key. default: %(default)s",
            default="group_id")
        parser.add_argument(
            "--map", help="Save resulting key to gorup_id map as a csv.")

    def map(self, data: dict) -> dict:
        args = self.args
        filepath: str = data[args.src_key]
        group_id = filepath2groupid(filepath)
        data[args.id_key] = group_id
        return data

    def finalize(self):
        args = self.args
        if args.map:
            with open(args.map, mode="w", newline="") as file:
                writer = csv.writer(file)
                for row in self.path2id:
                    writer.writerow(row)


class AddAnchor(DictMapper):

    @classmethod
    @property
    def name(cls) -> str:
        return 'anchor'

    @classmethod
    @property
    def help(cls) -> str:
        return 'Add anchor'

    @classmethod
    def add_arguments(cls, parser: ArgumentParser):
        pass

    def map(self, data: dict) -> dict:
        d_points = data['point']
        tl = np.array(sorted(d_points['TL'], key=lambda e: e[0][1]))[:, 0]
        tr = np.array(sorted(d_points['TR'], key=lambda e: e[0][1]))[:, 0]
        d_points['C7-TL'] = [[tl[0].tolist()]]
        d_points['C7-TR'] = [[tr[0].tolist()]]
        d_points['S-TL'] = [[tl[-1].tolist()]]
        d_points['S-TR'] = [[tr[-1].tolist()]]
        return data

    def finalize(self):
        pass


def main():
    parser = argparse.ArgumentParser(
        description=
        "Read ndjson of exp data from stdin and apply mapping functions.")
    sub_parsers = parser.add_subparsers(dest='command_name')
    for c in [InsertGroupId, AddAnchor]:
        p = sub_parsers.add_parser(c.name, help=c.help)
        c.add_arguments(p)
        p.set_defaults(new=c)

    if len(sys.argv) <= 1:
        parser.print_help()
        return 1

    # parse commands and args
    rest = sys.argv[1:]
    args = argparse.Namespace()
    commands = []
    while rest:
        args, rest = parser.parse_known_args(rest)
        commands.append(args.new(args))
        if not args.command_name:
            break

    # read jsons and apply commands
    for line in sys.stdin:
        data = json.loads(line)
        for c in commands:
            data = c.map(data)
        print(json.dumps(data, ensure_ascii=False))
    for c in commands:
        c.finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
