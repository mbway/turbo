#!/usr/bin/env python3

import sys
import pkgutil
sys.path.append('..')

with open('source/API.rst', 'r') as f:
    API_text = f.read()

import turbo

modules = [modname for importer, modname, ispkg in pkgutil.walk_packages(turbo.__path__, 'turbo.')]
modules.insert(0, 'turbo')

print('Missing Modules:')
any_missing = False
for m in modules:
    if m not in API_text:
        print('    {}'.format(m))
        any_missing = True
if not any_missing:
    print('none.')
