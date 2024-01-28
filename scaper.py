{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNTxEaWZvlUWAUKv0kjI6r9",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/qilinzho56/SP500/blob/main/scaper.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "3yWnZQrs1WyW"
      },
      "outputs": [],
      "source": [
        "import requests\n",
        "import lxml.html\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import re\n",
        "from datetime import date"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def headlines(headers, company_list, max_days):\n",
        "    \"\"\"\n",
        "    Scrape the stock page of each company and reformat the date information\n",
        "    from http://finviz.com.\n",
        "\n",
        "    Parameters\n",
        "    ----------\n",
        "    headers: HTTP header\n",
        "    company_list: a list of companies of interest\n",
        "    max_days: days that the user want to scrape util the current date\n",
        "\n",
        "    Returns\n",
        "    -------\n",
        "    df: a dataframe including the date, time, company, headline and and url link\n",
        "    \"\"\"\n",
        "    data = []\n",
        "    date_pattern = re.compile(r'[A-Za-z]+')\n",
        "\n",
        "    for company in company_list:\n",
        "        company_url = headers[\"Referer\"] + company\n",
        "        response = requests.get(company_url, headers=headers)\n",
        "        root = lxml.html.fromstring(response.text)\n",
        "        news_rows = root.xpath(\"//table[@id='news-table']/tr\")\n",
        "\n",
        "        cur_date = None\n",
        "        cur_time = None\n",
        "        days_visited = 0\n",
        "\n",
        "        for row in news_rows:\n",
        "            time_extract = \" \".join(row.xpath(\"./td/text()\"))\n",
        "            if re.match(date_pattern, time_extract.split()[0]):\n",
        "                if time_extract.split()[0] == \"Today\":\n",
        "                    cur_date = date.today()\n",
        "                    cur_date = cur_date.strftime(\"%b-%d-%y\")\n",
        "                    cur_time = time_extract.split()[1]\n",
        "                else:\n",
        "                    cur_date = time_extract.split()[0]\n",
        "                    cur_time = time_extract.split()[1]\n",
        "                days_visited += 1\n",
        "\n",
        "                if days_visited > max_days:\n",
        "                    break\n",
        "\n",
        "            headline = \" \".join(row.xpath(\".//a[@target='_blank']/text()\"))\n",
        "            url = row.xpath(\".//a[@target='_blank']/@href\")\n",
        "            if headline:\n",
        "                data.append([cur_date, cur_time, company, headline, url[0]])\n",
        "\n",
        "    return pd.DataFrame(data, columns=[\"Date\", \"Time\", \"Company\", \"Headline\", \"URL\"])\n"
      ],
      "metadata": {
        "id": "FkORZ1Sa6Iij"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}