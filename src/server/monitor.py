import datetime, psutil
import asyncio
from fastapi import FastAPI
from easycharts import ChartServer
from easyschedule import EasyScheduler

scheduler = EasyScheduler()
server = FastAPI()

every_minute = '* * * * *'

@server.on_event('startup')
async def setup():
    asyncio.create_task(scheduler.start())
    server.charts = await ChartServer.create(
        server,
        charts_db="charts_database",
        chart_prefix = '/mycharts'
    )

    server.charts.create_dataset(
        "test",
        labels=['a', 'b', 'c', 'd'],
        dataset=[1,2,3,4]
    )

    # set initial sync time
    label=datetime.datetime.now().isoformat()[11:19]
    server.charts.create_dataset(
        'cpu',
        labels=[label],
        dataset=[psutil.cpu_percent()]
    )
    server.charts.create_dataset(
        'mem',
        labels=[label],
        dataset=[psutil.virtual_memory().percent]
    )

@scheduler(schedule=every_minute)
async def resource_monitor():
    time_now=datetime.datetime.now().isoformat()[11:19]

    # updates CPU & MEM datasets with current time
    await server.charts.update_dataset(
        'cpu',
        label=time_now,
        data=psutil.cpu_percent()
    )
    await server.charts.update_dataset(
        'mem',
        label=time_now,
        data=psutil.virtual_memory().percent
    )