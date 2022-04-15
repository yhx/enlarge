#include <iostream>
#include <cmath>

using namespace std;

#define CONFIG_TICS_PER_MS 1000.0
#define CONFIG_TICS_PER_STEP 100 

const double TICS_PER_MS_DEFAULT = CONFIG_TICS_PER_MS;
const long TICS_PER_STEP_DEFAULT = CONFIG_TICS_PER_STEP;

long TICS_PER_STEP = TICS_PER_STEP_DEFAULT;
double TICS_PER_STEP_INV = 1. / static_cast< double >( TICS_PER_STEP );
long TICS_PER_STEP_RND = TICS_PER_STEP - 1;

double TICS_PER_MS = TICS_PER_MS_DEFAULT;
double MS_PER_TIC = 1 / TICS_PER_MS;

double MS_PER_STEP = TICS_PER_STEP / TICS_PER_MS;
double STEPS_PER_MS = 1 / MS_PER_STEP;

double
dround( double x )
{
  return std::floor( x );
}

void
set_resolution( double ms_per_step )
{
//   assert( ms_per_step > 0 );

  TICS_PER_STEP = static_cast< long >( dround( TICS_PER_MS * ms_per_step ) );
  TICS_PER_STEP_INV = 1. / static_cast< double >( TICS_PER_STEP );
  TICS_PER_STEP_RND = TICS_PER_STEP - 1;

  // Recalculate ms_per_step to be consistent with rounding above
  MS_PER_STEP = TICS_PER_STEP / TICS_PER_MS;
  STEPS_PER_MS = 1 / MS_PER_STEP;

//   const long max = compute_max();
//   LIM_MAX = +max;
//   LIM_MIN = -max;
}

int main() {

    set_resolution(1.0);

    cout << dround(1.0 * STEPS_PER_MS) << endl;
    cout << dround(2.0 * STEPS_PER_MS) << endl;
    cout << dround(3.0 * STEPS_PER_MS) << endl;
    cout << dround(4.0 * STEPS_PER_MS) << endl;
}
