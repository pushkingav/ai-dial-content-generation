import asyncio
from io import BytesIO
from pathlib import Path

from task._models.custom_content import Attachment, CustomContent
from task._utils.constants import API_KEY, DIAL_URL, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.bucket_client import DialBucketClient
from task._utils.model_client import DialModelClient
from task._models.message import Message
from task._models.role import Role


async def _put_image() -> Attachment:
    file_name = 'dialx-banner.png'
    image_path = Path(__file__).parent.parent.parent / file_name
    mime_type_png = 'image/png'

    async with DialBucketClient(api_key=API_KEY, base_url=DIAL_URL) as bucket_client:
        with open(image_path, "rb") as image_file:
            content = BytesIO(image_file.read())

        result = await bucket_client.put_file(
            name=file_name,
            mime_type=mime_type_png,
            content=content
        )

    url = result.get("url")
    return Attachment(title=file_name, url=url, type=mime_type_png)


def start() -> None:
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="gpt-4o",
        api_key=API_KEY
    )

    attachment = asyncio.run(_put_image())
    print("Uploaded attachment:", attachment)

    message = Message(
        role=Role.USER,
        content="What do you see on this picture?",
        custom_content=CustomContent(attachments=[attachment])
    )

    response = client.get_completion(messages=[message])
    print(response.content)


start()
