def crc8(data, len):

    crc = 0x00

    idx = 0
    while(len > 0):

        extract = data[idx]

        for tmp in range(8):
            sum = (crc ^ extract) & 0x01
            crc = crc >> 1
            if sum:
                crc ^= 0x8C
            extract = extract >> 1    

        len = len - 1
        idx = idx + 1

    return crc


