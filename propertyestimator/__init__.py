"""
propertyestimator
Property calculation toolkit from the Open Forcefield Consortium.
"""

# Make Python 2 and 3 imports work the same
# Safe to remove with Python 3-only code
from __future__ import absolute_import

import pkg_resources

from ._version import get_versions

# Load in any found plugins.
for entry_point in pkg_resources.iter_entry_points('propertyestimator.plugins'):

    try:
        entry_point.load()
    except ImportError as e:

        import traceback

        formatted_exception = traceback.format_exception(None, e, e.__traceback__)
        print(f'Could not load the {entry_point} plugin: {formatted_exception}')

# Handle versioneer
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
