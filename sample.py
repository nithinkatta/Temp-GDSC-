# This snippet has been automatically generated and should be regarded as a
# code template only.
# It will require modifications to work:
# - It may require correct/in-range values for request initialization.
# - It may require specifying regional endpoints when creating the service
#   client as shown in:
#   https://googleapis.dev/python/google-api-core/latest/client_options.html
from google.apps import meet_v2


async def sample_create_space():
    # Create a client
    client = meet_v2.SpacesServiceAsyncClient()

    # Initialize request argument(s)
    request = meet_v2.CreateSpaceRequest(
    )

    # Make the request
    response = await client.create_space(request=request)

    # Handle the response
    print(response)
sample_create_space()