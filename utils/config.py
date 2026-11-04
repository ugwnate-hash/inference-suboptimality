"""Load command-line defaults from a YAML config file.

Every script accepts ``--config <file.yaml>``. The values in the file replace the
script's argparse defaults, and anything given explicitly on the command line still
wins. Keys are flag names with ``_`` or ``-`` (``n_ais_dist`` == ``n-ais-dist``) and
may be grouped under sections, which are only there for readability. A script
ignores keys it doesn't define, so one file can serve all of them.
"""

import argparse
import sys


def _flatten(cfg, out):
    for key, value in cfg.items():
        if isinstance(value, dict):
            _flatten(value, out)
        else:
            key = str(key).replace('-', '_')
            if key in out:
                raise ValueError('config key %r appears more than once' % key)
            out[key] = value
    return out


def parse_args(parser, argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', '-c', type=str, default=None)
    known, _ = pre.parse_known_args(argv)

    parser.add_argument('--config', '-c', type=str, default=None,
                        help='YAML file with default values for these flags')

    if known.config is not None:
        import yaml  # only needed when a config file is used
        with open(known.config) as f:
            cfg = _flatten(yaml.safe_load(f) or {}, {})

        dests = {a.dest: a for a in parser._actions}
        applied = {}
        for key, value in cfg.items():
            action = dests.get(key)
            if action is None or key in ('help', 'config'):
                continue
            if action.choices is not None and value not in action.choices:
                parser.error('config %s: %r is not one of %s'
                             % (key, value, list(action.choices)))
            applied[key] = value
        parser.set_defaults(**applied)
        print('Defaults from %s: %s' % (known.config,
              ', '.join('%s=%s' % kv for kv in sorted(applied.items()))))
        sys.stdout.flush()

    return parser.parse_args(argv)
