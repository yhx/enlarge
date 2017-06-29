/* This program is writen by qp09.
 * usually just for fun.
 * Fri December 11 2015
 */

#include "IDPool.h"

IDPool sidPool;
IDPool nidPool;
IDPool sendTag;
IDPool recvTag;

IDPool::IDPool(int gid)
{
	this->id.setID(tmp);
}

ID IDPool::getID()
{
	ID tmp = id;
	id.advance();
	return tmp;
}

IDPool::~IDPool()
{
}