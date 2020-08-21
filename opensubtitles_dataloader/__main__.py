import click
from pathlib import Path
import path
import urllib.request
import shutil
import os.path as path
import os
import zipfile
import subprocess

@click.command()
@click.argument("language")
@click.option(
    "-l",
    "--location",
    help="Path to download the language to.",
    default=f"{Path.home()}/.cache/opensubtitles",
)
@click.option("-f", "--force", help="Path to download the language to.", is_flag=True)
def download(language, location, force=False):
    Path(location).mkdir(parents=True, exist_ok=True)
    print(f"{location}/{language}")
    if not path.exists(f"{location}/{language}") or force:
        try:
            if force:
                shutil.rmtree(f"{location}/{language}")
            Path(f"{location}/{language}").mkdir(parents=True, exist_ok=True)
            os.chdir(f"{location}/{language}")
            subprocess.check_call(f"wget -O {language}.zip http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/xml/{language}.zip".split())
            local_filename = f"{location}/{language}/{language}.zip"
            with open(local_filename, "rb") as fileobj:
                z = zipfile.ZipFile(fileobj)
                z.extractall(f"{location}/{language}")
                z.close()
            os.remove(local_filename)
        except KeyboardInterrupt:
            shutil.rmtree(f"{location}/{language}")
        except subprocess.CalledProcessError:
            shutil.rmtree(f"{location}/{language}")
    else:
        click.echo(f"Language {language} already downloaded.")
