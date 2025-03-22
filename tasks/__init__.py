"""All tasks are defined in this package."""

from invoke import Collection

from tasks import checks, setup


ns = Collection()
ns.add_collection(checks)
ns.add_collection(setup)
