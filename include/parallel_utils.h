#pragma once

#include <functional>

/*!
 * Run f with size threads
 * @param size : thread count
 * @param f : thread function
 */
void MultiThreadRun(int size, std::function<void(int, int)> f);

/*!
 * Run f with size processes
 * @param size : process count
 * @param f : process function
 */
void MultiProcessRun(int size, std::function<void(int, int)> f);
