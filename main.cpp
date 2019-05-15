#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include "dds.h"
#include "csv.h"

float const MATH_PI = 3.14159f;

uint32_t ReverseBits( uint32_t v )
{
    v = ( ( v >> 1 ) & 0x55555555 ) | ( ( v & 0x55555555 ) << 1 );
    v = ( ( v >> 2 ) & 0x33333333 ) | ( ( v & 0x33333333 ) << 2 );
    v = ( ( v >> 4 ) & 0x0F0F0F0F ) | ( ( v & 0x0F0F0F0F ) << 4 );
    v = ( ( v >> 8 ) & 0x00FF00FF ) | ( ( v & 0x00FF00FF ) << 8 );
    v = (   v >> 16               ) | (   v                << 16 );
    return v;
}

union FP32
{
    unsigned    u;
    float       f;
};

// https://gist.github.com/rygorous/2156668
uint16_t FloatToHalf( float ff )
{
    FP32 f32infty = { 255 << 23 };
    FP32 f16infty = { 31 << 23 };
    FP32 magic = { 15 << 23 };
    unsigned sign_mask = 0x80000000u;
    unsigned round_mask = ~0xfffu; 

    uint16_t o = 0;
    FP32 f;
    f.f = ff;

    unsigned sign = f.u & sign_mask;
    f.u ^= sign;

    // NOTE all the integer compares in this function can be safely
    // compiled into signed compares since all operands are below
    // 0x80000000. Important if you want fast straight SSE2 code
    // (since there's no unsigned PCMPGTD).

    if (f.u >= f32infty.u) // Inf or NaN (all exponent bits set)
        o = (f.u > f32infty.u) ? 0x7e00 : 0x7c00; // NaN->qNaN and Inf->Inf
    else // (De)normalized number or zero
    {
        f.u &= round_mask;
        f.f *= magic.f;
        f.u -= round_mask;
        if (f.u > f16infty.u) f.u = f16infty.u; // Clamp to signed infinity if overflowed

        o = uint16_t( f.u >> 13 ); // Take the bits!
    }

    o |= sign >> 16;
    return o;
}

float Vis( float roughness, float ndotv, float ndotl )
{
    // GSmith correlated
    float m     = roughness * roughness;
    float m2    = m * m;
    float visV  = ndotl * sqrt( ndotv * ( ndotv - ndotv * m2 ) + m2 );
    float visL  = ndotv * sqrt( ndotl * ( ndotl - ndotl * m2 ) + m2 );
    return 0.5f / ( visV + visL );
}

float VisBeckmann(float roughness, float ndotv, float ndotl)
{
    // Smith Beckmann Vis used for Blinn Phong
    float a = roughness;

    // http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
    //float k = a * sqrt(2.0f / 3.14159265f);

    // https://learnopengl.com/PBR/Theory // ??
    float k = (a * a) / 2.0f; // ??

    float G_V = ndotv * (1.0f - k) + k;
    float G_L = ndotl * (1.0f - k) + k;

    float G = (ndotv * ndotl) / (G_V * G_L);

    float vis = G / (4.0f * ndotl * ndotv);

    return vis;
}

// Returns cosTheta
float ImportanceSampleGGX(float xi, float m2)
{
    return sqrt((1.0f - xi) / (1.0f + (m2 - 1.0f) * xi));
}

// Returns cosTheta
float ImportanceSampleBlinnPhong(float xi, float specPow)
{
    return pow(xi, 1.0f / (specPow + 2.0f));
}

float BlinnPowerToBeckmannRoughness(float s)
{
    return sqrt(2.0f / (s + 2.0f));
}

#define SAMPLE_BLINN_PHONG 1

int main()
{
    unsigned const LUT_WIDTH  = 128;
    unsigned const LUT_HEIGHT = 128;
    unsigned const sampleNum  = 512;

    float lutDataRGBA32F[ LUT_WIDTH * LUT_HEIGHT * 4 ];
    uint16_t lutDataRG16F[ LUT_WIDTH * LUT_HEIGHT * 2 ];

    for ( unsigned y = 0; y < LUT_HEIGHT; ++y )
    {
        float const ndotv = ( y + 0.5f ) / LUT_HEIGHT;

        for ( unsigned x = 0; x < LUT_WIDTH; ++x )
        {
#ifdef SAMPLE_BLINN_PHONG
            float const gloss = ( x + 0.5f ) / LUT_WIDTH;
            float const p = exp2(13.0f * gloss);
            float const roughness = BlinnPowerToBeckmannRoughness(p);
#else
            float const roughness = ( x + 0.5f ) / LUT_WIDTH;
            float const m = roughness * roughness;
            float const m2 = m * m;
#endif

            float const vx = sqrtf( 1.0f - ndotv * ndotv );
            float const vy = 0.0f;
            float const vz = ndotv;

            float scale = 0.0f;
            float bias  = 0.0f;

            for ( unsigned i = 0; i < sampleNum; ++i )
            {
                float const e1 = (float) i / sampleNum;
                float const e2 = (float) ( (double) ReverseBits( i ) / (double) 0x100000000LL );

                float const phi         = 2.0f * MATH_PI * e1;
                float const cosPhi      = cosf( phi );
                float const sinPhi      = sinf( phi );

#ifdef SAMPLE_BLINN_PHONG
                float const cosTheta = ImportanceSampleBlinnPhong(e2, p);
#else
                float const cosTheta = ImportanceSampleGGX(e2, m2);
#endif

                float const sinTheta    = sqrtf( 1.0f - cosTheta * cosTheta );

                float const hx  = sinTheta * cosf( phi );
                float const hy  = sinTheta * sinf( phi );
                float const hz  = cosTheta;

                float const vdh = vx * hx + vy * hy + vz * hz;
                float const lx  = 2.0f * vdh * hx - vx;
                float const ly  = 2.0f * vdh * hy - vy;
                float const lz  = 2.0f * vdh * hz - vz;

                float const ndotl = std::max( lz,  0.0f );
                float const ndoth = std::max( hz,  0.0f );
                float const vdoth = std::max( vdh, 0.0f );

                if ( ndotl > 0.0f )
                {
#ifdef SAMPLE_BLINN_PHONG
                    float const vis = VisBeckmann(roughness, ndotv, ndotl);

#else
                    float const vis = Vis(roughness, ndotv, ndotl);
#endif

                    float const ndotlVisPDF = ndotl * vis * (4.0f * vdoth / ndoth);
                    float const fresnel = powf(1.0f - vdoth, 5.0f);

                    scale += ndotlVisPDF * ( 1.0f - fresnel );
                    bias  += ndotlVisPDF * fresnel;
                }
            }
            scale /= sampleNum;
            bias  /= sampleNum;


            //if (scale > 1.f)
            //    printf("sacle warning: %.2f - %.2f, %.3f\n", ndotv, gloss, scale);
            //if (bias > 1.f)
            //    printf("bias warning: %.2f - %.2f, %.3f\n", ndotv, gloss, bias);


            lutDataRGBA32F[ x * 4 + y * LUT_WIDTH * 4 + 0 ] = scale;
            lutDataRGBA32F[ x * 4 + y * LUT_WIDTH * 4 + 1 ] = bias;
            lutDataRGBA32F[ x * 4 + y * LUT_WIDTH * 4 + 2 ] = 0.0f;
            lutDataRGBA32F[ x * 4 + y * LUT_WIDTH * 4 + 3 ] = 0.0f;

            lutDataRG16F[ x * 2 + y * LUT_WIDTH * 2 + 0 ] = FloatToHalf( scale );
            lutDataRG16F[ x * 2 + y * LUT_WIDTH * 2 + 1 ] = FloatToHalf( bias );
        }
    }   

    SaveDDS( "integrateDFG_RGBA32F.dds", DDS_FORMAT_R32G32B32A32_FLOAT, 16, LUT_WIDTH, LUT_HEIGHT, lutDataRGBA32F );
    SaveDDS( "integrateDFG_RG16F.dds", DDS_FORMAT_R16G16_FLOAT, 4, LUT_WIDTH, LUT_HEIGHT, lutDataRG16F );
    SaveCSV( "ndotv.csv", LUT_WIDTH );
    SaveCSV( "gloss.csv", LUT_HEIGHT );
    SaveCSV( "scale.csv", lutDataRGBA32F, LUT_WIDTH, LUT_HEIGHT, 0 );
    SaveCSV( "bias.csv",  lutDataRGBA32F, LUT_WIDTH, LUT_HEIGHT, 1 );

    return 0;
}