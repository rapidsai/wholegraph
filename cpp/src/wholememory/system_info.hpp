#pragma once

bool DevAttrPagebleMemoryAccess();

bool DeviceCanAccessPeer(int peer_device);

bool DevicesCanAccessP2P(const int* dev_ids, int count);

int GetCudaCompCap();

const char* GetCPUArch();

bool SupportMNNVL();

bool SupportEGM();

bool SupportMNNVLForEGM();
