import logging
import socket
import uuid

from elasticsearch import Elasticsearch

from config.config import ES_HOST, ES_PORT, IS_USER, USER_NAME, PASSWORD, INDEX, DOC_TYPE, VERSION


class Elastic(logging.Handler):
    def __init__(self):
        if IS_USER:
            self.es = Elasticsearch([{'host': ES_HOST, 'port': ES_PORT}], http_auth=(USER_NAME, PASSWORD), timeout=3600)
        else:
            self.es = Elasticsearch([{'host': ES_HOST, 'port': ES_PORT}], timeout=3600)
        self.ip = self.get_host_ip()
        self.version = VERSION
        logging.Handler.__init__(self)

    def get_host_ip(self):
        """
        查询本机ip地址
        :return: ip
        """
        # 获取本机电脑名
        myname = socket.gethostname()
        # 获取本机ip
        ip = socket.gethostbyname(myname)
        return ip

    def emit(self, record):
        self.es.indices.create(index=INDEX, ignore=400)
        msg = '[{}] [{}] [{}] [{}] [{}] {}'.format(record.asctime, record.name, self.ip, self.version, record.levelname,
                                                   record.msg)
        data = {'log': msg}
        self.es.index(index=INDEX, doc_type=DOC_TYPE, id=uuid.uuid1(), body=data)
