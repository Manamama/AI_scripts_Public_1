# Sydney Client with Image Support

This project is a Python script that uses the Sydney Client to interact with OpenAI's Sydney API. It supports sending image prompts by uploading local files or providing URLs. 

## Features

- Interact with the Sydney API using text prompts.
- Send image prompts by uploading a local file or providing a URL.
- Display responses from Sydney in the terminal with Markdown formatting.

## Dependencies

- Python 3.7 or higher
- `rich` library for rendering Markdown in the terminal
- `sydney-py` library [https://github.com/vsakkas/sydney.py] for interacting with the Sydney aka Bing's API
- `tkinter` library for showing a file picker dialog

## Usage

1. Run the script in your terminal:

```bash
python main.py
```
Enter a text prompt to send to Sydney. For example:
`User: Tell me a joke.`

If you want to send an image prompt, start your input with !image, followed by the URL or local file path of the image, and then the text prompt. For example:
User: !image https://example.com/image.jpg What animal is this?

If you don’t provide a URL after !image, the script will show a file picker dialog for you to select a local file.

The script will display Sydney’s response in the terminal with Markdown formatting.

## Note
This script is an example of how to use the Sydney API and is not intended for production use.
