#include "NetWork.h"
#include <SFML/Graphics.hpp>
using namespace sf;


struct data_pix {
    double* pixels;
    int dig;
};

void readImage() {
    Image img;
    bool flag = true;
    flag = img.loadFromFile("digits/five.png");
    if (!flag) {
        cout << "Error loading the file";
        system("pause");
    }
    ofstream fout;
    fout.open("lib.txt");

    if (!fout.is_open()) {
        cout << "Error open the file" << endl;
        system("pause");
    }

    int w = img.getSize().x;
    int h = img.getSize().y;
    cout << "SIZE OF IMAGE " << w << " " << h << endl;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            if (j == w - 1) {
                // чтобы не было пустого символа
                fout << fixed << setprecision(3) << 1.0 - img.getPixel(j, i).r / 255.0;
                continue;
            }
            fout << fixed << setprecision(3) << 1.0 - img.getPixel(j, i).r / 255.0 << " ";
        }
        if (i == w - 1) continue;
        fout << endl;
    }
    fout.close();
}

void full_input(data_pix* data_info,int size_input) {
    ifstream fin;
    fin.open("123.txt");
    if (!fin.is_open()) {
        cout << "Error open the file" << endl;
        system("pause");
    }
    double tmp;
    while (!fin.eof()) {
        for (int k = 0; k < 4; k++) {
            fin >> data_info[k].dig;
            cout << "DIGIT: " << data_info[k].dig << endl;
            for (int i = 0; i < size_input; i++) {
                fin >> data_info[k].pixels[i];
                if (i % 28 == 0) cout << endl;
                cout << data_info[k].pixels[i] << " ";
            }
            cout << endl;
        }
    }
    fin.close();
}


int main()
{
    const int size_input = 784;
    vector <double> input;

    const int n = 4;
    int size[n] = { size_input,50,20,10 }; // размерность нейросети
    //Выделение памяти под библиотеку пикселей и цифр
    data_pix* data_info = new data_pix[size[n - 1]];
    for (int i = 0; i < size[n-1]; i++) {
        data_info[i].pixels = new double[size_input];
    }
    //ifstream fin;
    //fin.open("123.txt");
    //if (!fin.is_open()) {
    //    cout << "Error open the file" << endl;
    //    system("pause");
    //}
    //double tmp;
    //for (int i = 0; i < 2000; i++) {
    //    fin >> tmp;
    //    if (i % 28 == 0) cout << endl;
    //    cout << tmp << " ";
    //}
    //fin.close();

    readImage();

    full_input(data_info, size_input);

    NetWork hi{};
    hi.SetLayers(n, size);
    hi.SetInput(data_info[0].pixels);
    hi.forward_feed();


    return 0;
}
