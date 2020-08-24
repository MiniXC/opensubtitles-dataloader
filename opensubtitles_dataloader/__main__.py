import click
from pathlib import Path
import path
import urllib.request
import shutil
import os.path as path
import os
import gzip
import subprocess


@click.command()
@click.argument("language")
@click.option(
    "-l",
    "--location",
    help="Path to download the language to.",
    default=f"{Path.home()}/.cache/opensubtitles",
)
@click.option("-f", "--force", help="Force redownload.", is_flag=True)
@click.option(
    "-t", "--token", help="Download the tokenized version of the dataset.", is_flag=True
)
def download(language, location, force=False, token=False):
    Path(location).mkdir(parents=True, exist_ok=True)
    print(f"{location}/{language}")
    if not path.exists(f"{location}/{language}") or force:
        try:
            if force:
                shutil.rmtree(f"{location}/{language}")
            Path(f"{location}/{language}").mkdir(parents=True, exist_ok=True)
            os.chdir(f"{location}/{language}")
            if token:
                download_path = f"http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.{language}.gz"
            else:
                download_path = f"http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.{language}.gz"
            subprocess.check_call(f"wget -O {language}.txt.gz {download_path}".split())
            local_filename = f"{location}/{language}/{language}.txt.gz"
            with gzip.open(local_filename, "rb") as f_in:
                with open(local_filename.replace(".gz", ""), "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(local_filename)
        except KeyboardInterrupt:
            shutil.rmtree(f"{location}/{language}")
        except subprocess.CalledProcessError:
            shutil.rmtree(f"{location}/{language}")
    else:
        click.echo(f"Language {language} already downloaded.")
