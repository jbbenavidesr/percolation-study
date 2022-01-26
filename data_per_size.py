import os

from faunadb import query as q
from faunadb.objects import Ref
from faunadb.client import FaunaClient

print(os.environ.get("FAUNA_SECRET"))

client = FaunaClient(
    secret=os.environ.get("FAUNA_SECRET"),
)

data = client.query(q.get(q.ref(q.collection("jigsawData"), "318233311471206991")))

print(data)
