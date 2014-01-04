'''
Created on Oct 31, 2013

@author: c3h3
'''

from __future__ import absolute_import

from celery import Celery

celery = Celery('connecting_engine.celery',
                broker='amqp://',
                backend="mongodb://localhost:27017/bows_celery",
                include=['bows_celery.tasks'])



if __name__ == '__main__':
    celery.start()
