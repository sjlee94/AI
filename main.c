#include<stdio.h>
#include "function.h"

void main() {
    FILE *fp = NULL;
    int i, j, k, l;


    //          H  W  I  O
    double L1_W[3][3][3][6];
    double L1_B[6];
    double L2_W[3][3][6][9];
    double L2_B[9];

    double L5_W[900][16];
    double L5_B[16];
    double L6_W[16][3];
    double L6_B[3];

    //         W   H   CH
    int   img[24][24][3];

    //      Activation map
    double ***L1_output = NULL;
    double ***L2_output = NULL;
    double ***L3_output = NULL;
    double   *L4_output = NULL;
    double   *L5_output = NULL;
    double   *L6_output = NULL;

    // READ parameter
    fp = fopen("weights.bin", "rb");

    fread( L1_W, sizeof(double), 3*3*3*6, fp );
    fread( L1_B, sizeof(double), 6, fp );
    fread( L2_W, sizeof(double), 3*3*6*9, fp );
    fread( L2_B, sizeof(double), 9, fp );
    fread( L5_W, sizeof(double), 900*16, fp );
    fread( L5_B, sizeof(double), 16, fp );
    fread( L6_W, sizeof(double), 16*3, fp );
    fread( L6_B, sizeof(double), 3, fp );
    fclose(fp);
    
    
    // READ image
//    fp =  fopen("circle.txt", "r");
    fp =  fopen("rectangle.txt", "r");
//    fp =  fopen("triangle.txt", "r");
    for( i=0 ; i<3 ; i++ ) {
        for( j=0 ; j<24 ; j++ ) {
            for( k=0 ; k<24 ; k++ ) {
                fscanf(fp, "%d", &img[j][k][i]);
            }
        }
    }
    fclose(fp);

/*
    for( i=0 ; i<3 ; i++ ) {
        for( j=0 ; j<3 ; j++ ) {
            for( k=0 ; k<3 ; k++ ) {
                for( l=0 ; i<4 ; l++ ) {
                    

    for( i=0 ; i<3 ; i++ ) {
        for( j=0 ; j<24 ; j++ ) {
            for( k=0 ; k<24 ; k++ ) {
                printf("%3d ", img[j][k][i]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
*/
    
    L1_output = conv2d_1( img, L1_W, L1_B );

 /*
    fp = fopen("act_L1.txt", "w");
    for( k=0 ; k<4 ; k++ ) {
        for( i=0 ; i<22 ; i++ ) {
            for( j=0 ; j<22 ; j++ ) {
                fprintf(fp, "%f\t", L1_output[i][j][k]);
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n\n");
    }
    fclose(fp);
*/

    L2_output = conv2d_2( L1_output, L2_W, L2_B );
/*    
    fp = fopen("act_L2.txt", "w");
    for( k=0 ; k<8 ; k++ ) {
        for( i=0 ; i<20 ; i++ ) {
            for( j=0 ; j<20 ; j++ ) {
                fprintf(fp, "%f\t", L2_output[i][j][k]);
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n\n");
    }
    fclose(fp);
*/
    L3_output = max_pooling2d( L2_output );
/*
    fp = fopen("act_L3.txt", "w");
    for( k=0 ; k<8 ; k++ ) {
        for( i=0 ; i<10 ; i++ ) {
            for( j=0 ; j<10 ; j++ ) {
                fprintf(fp, "%f\t", L3_output[i][j][k]);
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n\n");
    }
    fclose(fp);
*/
    L4_output = flatten( L3_output );
/*    
    fp = fopen("act_L4.txt", "w");
    for( i=0 ; i<800 ;i++ ) {
        fprintf(fp, "%f\n", L4_output[i]);
    }
    fclose(fp);
*/
    L5_output = dense_1d( L4_output, L5_W, L5_B );
/*
    fp = fopen("act_L5.txt", "w");
    for( i=0 ; i<6 ;i++ ) {
        fprintf(fp, "%f\n", L5_output[i]);
    }
    fclose(fp);
*/
    L6_output = dense_2d( L5_output, L6_W, L6_B );
/*
    fp = fopen("act_L6.txt", "w");
    for( i=0 ; i<3 ;i++ ) {
        fprintf(fp, "%f\n", L6_output[i]);
    }
    fclose(fp);
*/
    printf("circle: 0, rectangle: 1, triangle: 2  \n");
    for( i=0 ; i<3 ; i++ )
        printf("%f\t", L6_output[i]);
    printf("\n");

}

