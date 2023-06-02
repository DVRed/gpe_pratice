import requests
from datetime import date
from typing import Mapping
import asyncio
import aiohttp
from loguru import logger

urls: Mapping[int, str] = {
    2015: 'https://www.nationalgas.com/document/69706/download',
    2016: 'https://www.nationalgas.com/gas-transmission/document/69701/download',
    2017: 'https://www.nationalgas.com/document/94086/download',
    2018: 'https://www.nationalgas.com/document/123221/download',
    2019: 'https://www.nationalgas.com/document/128376/download',
    2020: 'https://www.nationalgas.com/document/133121/download',
    2021: 'https://www.nationalgas.com/document/137111/download',
    2022: 'https://www.nationalgas.com/gas-transmission/document/140886/download',
    2023: 'https://www.nationalgas.com/document/130966/download'
    }


async def fetch(session: aiohttp.ClientSession, year: int) -> tuple[bytes, int]:
    async with session.get(url=urls[year], ssl=False) as result:
        return await result.read(), year


async def main() -> dict:
    results = dict()
    async with aiohttp.ClientSession() as session:
        pending = [
            asyncio.create_task(fetch(session, year)) for year in list(urls.keys())
        ]
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for done_task in done:
                if not done_task.exception():
                    result, year = await done_task
                    results[year] = result
                    logger.success(f'Received data for {year} year')
                else:
                    logger.error(f'Error when receiving data for {year} year')
    return results


def save_historical_data(file_name: str) -> None:
    result = asyncio.run(main())
    for key, value in result.items():
        with open(f'{file_name}_{key}.xls', 'wb') as f:
            f.write(value)


def save_current_data(file_name: str) -> None:
    response = requests.get(url=urls[date.today().year])
    with open(f'{file_name}.xls', 'wb') as f:
        f.write(response.content)
        
def save_current_data(file_name: str) -> None:
    response = requests.get(url=urls[date.today().year])
    with open(f'{file_name}.xls', 'wb') as f:
        f.write(response.content)


if __name__ == '__main__':
    current_date = date.today().strftime('%d.%m.%Y')
    folder = 'parsed_data'
    # save_current_data(f'{folder}/NG_{current_date}')
    save_historical_data(f'{folder}/NG')
