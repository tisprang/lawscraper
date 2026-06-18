import asyncio
import importlib
import os
from pathlib import Path

import ms_fl_scraper

ms_fl_scraper = importlib.reload(ms_fl_scraper)

output_file = Path('state_codes/California/l.jsonl')
if output_file.exists():
    output_file.unlink()
    print(f'removed {output_file}')

if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

async def main():
    await ms_fl_scraper.scrape_section_url_async(
        section_url='https://codes.findlaw.com/ca/government-code/',
        state='California',
        output_file=str(output_file),
        require_complete_tree=True,
    )

asyncio.run(main())
