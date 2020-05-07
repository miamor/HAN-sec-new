with open('list_api.txt', 'r') as f:
    list_api = [line.strip().lower() for line in f.readlines()]


iapis = ['ntduplicateobject', 'deviceiocontrol', 'movefilewithprogresstransactedw', 'openservicea', 'ntquerysysteminformation', 'ntsetvaluekey', 'wnetgetprovidernamew', 'ntsetinformationfile', 'ntcreateprocessex', 'ntcreatekey', 'rtlcreateuserprocess', 'movefilewithprogressw', 'cryptexportkey', 'openservicew', 'ntopenprocess', 'controlservice', 'cryptencrypt', 'ntterminateprocess', 'ntclose', 'getadaptersaddresses', 'crypthashdata', 'regqueryvalueexw', 'getclipboarddata', 'process32nextw', 'regsetvalueexa', 'createservicea', 'regopenkeyexw', 'ntdelayexecution', 'ntdeviceiocontrolfile', 'setclipboardviewer', 'ntallocatevirtualmemory', 'readprocessmemory', 'regopenkeyexa', 'shellexecuteexw', 'ntwritefile', 'ldrgetdllhandle', 'cryptgenkey', 'createservicew', 'getcomputernamew', 'regqueryvalueexa', 'ntopenfile', 'internetreadfile', 'obtainuseragentstring', 'urldownloadtocachefilew', 'getusernamea', 'ntcreatefile', 'addclipboardformatlistener', 'getcomputernamea', 'ntloaddriver', 'ntcreateprocess', 'ntprotectvirtualmemory', 'enumservicesstatusa', 'regsetvalueexw', 'internetsetoptiona', 'setwindowshookexa', 'ldrgetprocedureaddress', 'setwindowshookexw', 'enumservicesstatusw', 'process32firstw', 'setfileattributesw', 'internetopena', 'ldrloaddll', 'ntcreateuserprocess', 'internetopenw', 'createprocessinternalw', 'urldownloadtofilew']

# print('list_api', list_api)
count = 0
for iapi in iapis:
    # print(iapi)
    if iapi in list_api:
        print(iapi)
        count += 1

print(count)

# file 8
#misc 3
#network 7
#process 13
#reg 8
# service 7
# system 8
# crypt 4