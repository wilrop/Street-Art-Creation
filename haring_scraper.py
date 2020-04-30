# Code modified from WikiArt scraper
import time
import argparse
import urllib
import urllib.request
from bs4 import BeautifulSoup
from os import path


def get_image_link(number):
    try:
        url = "http://www.haring.com/!/art-work/%d" % number
        soup = BeautifulSoup(urllib.request.urlopen(url), "lxml")
        img = soup.find('img', {"class": "alignleft wp-post-image"})
        link = img.get('src')
        return link
    except Exception as e:
        print('failed to scrape %s' % url, e)


def downloader(link, output_dir):
    num_downloaded = 0
    filepath = link.split('/')
    savepath = path.join(output_dir, filepath[-1])
    try:
        time.sleep(0.2)  # try not to get a 403
        urllib.request.urlretrieve(link, savepath)
        num_downloaded += 1
        if num_downloaded % 100 == 0:
            print('downloaded number %d / %d...' % (num_downloaded, num_images))
    except Exception as e:
        print("failed downloading " + str(link), e)


def main(output_dir):
    print('gathering links to images... this may take a few minutes')
    numbers = list(range(1, 888))

    links = []
    for i in numbers:
        link = get_image_link(i)
        if link is not None:
            links.append(link)
        if len(links) % 10 == 0:
            print("Gathered links to " + str(len(links)) + " images")

    num_images = len(links)

    print('attempting to download %d images' % num_images)
    for idx, link in enumerate(links):
        downloader(link, output_dir)
        if (idx + 1) % 10 == 0:
            print("Downloaded " + str(idx + 1) + " items")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", help="where to put output files")
    args = parser.parse_args()
    output_dir = args.output_dir

    main(output_dir)
