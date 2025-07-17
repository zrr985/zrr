#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "netWorkBase.hpp"
#include "tcpUdp.hpp"



struct data
{
    char buf[64];
};

std::atomic<int> disFlag = 0;

void serverCallBack(netWorkBase* nethandler, void* buf, int size, sockaddr_in client_addr)
{
    struct ipPort dest;
    char client_ip[INET_ADDRSTRLEN];
    char myBuf[128];
    const char* reply = "Server: ";
    if(size > 0)
    {
        sprintf(myBuf, reply);
        // print(INFO, "size = %d\n", size);
        memcpy(myBuf + strlen(reply), buf, size);
        inet_ntop(AF_INET, &(client_addr.sin_addr), client_ip, INET_ADDRSTRLEN);
        dest.ip = client_ip;
        dest.port = ntohs(client_addr.sin_port);
        print(INFO, "Client Ip: %s, Port: %d\n", dest.ip, dest.port);
        nethandler->sendData(myBuf, strlen(reply) + size, dest);
    }
    else
    {
        print(INFO, "Client disconnect\n");
        disFlag++;
    }
}

int main()
{
    struct ipPort source = {
        .ip = "127.0.0.1",
        .port = 6000,
    };
    struct ipPort dest = {
        .ip = "127.0.0.1",
        .port = 6001,
    };
    struct data myData;
    netWorkBase* udpHandler = new tcpTool(source);
    if(udpHandler->createNet(sizeof(struct data)) == SCOKET_BIND_FAIL)
    {
        return 0;
    }
    udpHandler->startRecvThread(C_S_TYPE::SERVER, serverCallBack);
    // print(DEBUG, "Hello\n");
    // print(ERROR, "Hello\n");
    print(INFO, "Hello\n");
    while(1)
    {
        // int res = udpHandler->getData(&myData, sizeof(myData));
        // if(res > 0)
        // {
        //     print(INFO, "data str: %s\n", myData.buf);
        //     break;
        // }
        if(disFlag >= 3)
        {
            break;
        //     disFlag = false;
        //     udpHandler->endRecvThread();
        //     udpHandler->startRecvThread(C_S_TYPE::SERVER, serverCallBack);
        }
    }
    udpHandler->endRecvThread();
    udpHandler->destroryNet();
    if(udpHandler)
    {
        delete udpHandler;
        udpHandler = nullptr;
    }
    return 0;
}


// UDP
// int main()
// {
//     struct ipPort source = {
//         .ip = "127.0.0.1",
//         .port = 6000,
//     };
//     struct ipPort dest = {
//         .ip = "127.0.0.1",
//         .port = 6001,
//     };
//     struct data myData;
//     netWorkBase* udpHandler = new udpTool(source);
//     udpHandler->createNet(sizeof(struct data));
//     udpHandler->startRecvThread(C_S_TYPE::CLIENT, clientCallBack);
//     // print(DEBUG, "Hello\n");
//     // print(ERROR, "Hello\n");
//     print(INFO, "Hello\n");
//     while(1)
//     {
//         // strcpy(myData.buf, "123\n\0");
//         // udpHandler->sendData(&myData, strlen(myData.buf), dest);
        
//         int res = udpHandler->getData(&myData, sizeof(myData));
//         if(res > 0)
//         {
//             print(INFO, "data str: %s\n", myData.buf);
//             udpHandler->endRecvThread();
//             udpHandler->destroryNet();
//             break;
//         }
//     }
//     if(udpHandler)
//     {
//         delete udpHandler;
//         udpHandler = nullptr;
//     }
//     return 0;
// }