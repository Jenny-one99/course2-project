import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']
    
    #!aws s3 cp s3://{bucket}/{key} ./tmp/image.png
    #s3://{bucket}/models/image_model
    
    # Download the data from s3 to /tmp/image.png
    #img_1 = s3.Bucket(bucket).download_file(key)
    #plt.imsave("/tmp/image.png", img_1)
    #s3 = boto3.client('s3')
    s3.download_file(bucket, key, '/tmp/image.png')

    #with open('/tmp/image.png', 'wb') as f:
    #    s3.download_fileobj(bucket, key, f)
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        
        
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        
    }


import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

ENDPOINT='image-classification-2021-11-07-08-10-47-194'

def lambda_handler(event, context):

    key = event['s3_key']
    bucket = event['s3_bucket']
    image_data=event['image_data']
    
    # Decode the image data
    image = base64.b64decode(event['image_data'])

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(
        ENDPOINT, 
        sagemaker_session=sagemaker.Session())

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction:
    inferences = predictor.predict(image)
    
    # We return the data back to the Step Function    
    data= inferences.decode('utf-8')
    
    return {
        
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": data
            
    }








import json


THRESHOLD = 0.8


def lambda_handler(event, context):
    
    key = event['s3_key']
    bucket = event['s3_bucket']
    image_data=event['image_data']
    data=event["inferences"]
    
    # Grab the inferences from the event
    inferences =event["inferences"]
    
    n1=float(inferences[1:7])
    n2=1-n1
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = (n1>THRESHOLD)or(n2>THRESHOLD)
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": data
    }