import time
from pinecone import ServerlessSpec
from pinecone import Pinecone
from src.config.appconfig import PINECONE_API_KEY, PINECONE_INDEX



def configure_pinecone_index():
    # configure client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX == pc.list_indexes()[0]["name"]:
        print("VectorDatabase :: Ready!")
    else:
        
        spec = ServerlessSpec(cloud='aws', region='us-east-1')
        # create a new index
        pc.create_index(
            PINECONE_INDEX,
            dimension=1024,  # dimensionality of mixedbread large
            metric='dotproduct',
            spec=spec
        )

        # wait for index to be initialized
        while not pc.describe_index(PINECONE_INDEX).status['ready']:
            time.sleep(1)

        print("VectorDatabase :: Ready!")