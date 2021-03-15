#include "NetWork.h"
#include <SFML/Graphics.hpp>
using namespace sf;

const int n = 3;
const int examples = 60000;
const int input_n = 784;

struct data_info {
    double* pixels;
    double digit;
};
void read_data(data_info* data,int examples,int input_n) {
    ifstream fin;
    fin.open("lib_MNIST.txt");
    if (!fin.is_open()) {
        cout << "Error reading the file LIB" << endl;
        system("pause");
    }
    for (int i = 0; i < examples; i++) {
        fin >> data[i].digit;
        for (int j = 0; j < input_n; j++) fin >> data[i].pixels[j];
    }
    fin.close();
}
void read_test(double* input, int input_n) {
    ifstream fin;
    fin.open("test.txt");
    if (!fin.is_open()) {
        cout << "Error reading the file TEST" << endl;
        system("pause");
    }
    for (int j = 0; j < input_n; j++) fin >> input[j];
    fin.close();
}
void ShowWindow() {
    bool drawing = false;
    RenderWindow window(VideoMode(140, 140), L"Drawing here", Style::Default);

    RenderTexture canvas;
    canvas.create(140, 140);
    canvas.clear(Color::White);

    Sprite sprite,spr;
    sprite.setTexture(canvas.getTexture());
    CircleShape brush(5);
    brush.setFillColor(Color(10,10,10));
    brush.setOutlineColor(Color(40, 40, 40));
    brush.setOutlineThickness(1);

    Image img;
    Texture texture;
    Vector2u vc;
    vc.x = 28;
    vc.y = 28;
    Image img1;
    ofstream fout("test.txt");
    int w, h;
    while (window.isOpen()) {
        Event event;
        while (window.pollEvent(event)) {
            switch (event.type) {
            case Event::Closed:
                window.close();
                break;
            case sf::Event::KeyPressed:
                switch (event.key.code) {
                case Keyboard::C:
                    canvas.clear(Color::White);
                    canvas.display();
                    break;
                case sf::Keyboard::G:
                    sprite.scale(0.2, 0.2);
                    break;
                case Keyboard::V:
                    texture.create(window.getSize().x, window.getSize().y);
                    texture.update(window);
                    img = texture.copyToImage();
                    w = img.getSize().x-112;
                    h = img.getSize().y-112;
                    cout << w << " " << h << endl;
                    for (int i = 0; i < h; i++) {
                        for (int j = 0; j < w; j++) fout << 1.0 - img.getPixel(j, i).r / 255. << " ";
                        fout << endl;
                    }
                    cout << "Saved" << endl;
                    fout.close();
                    break;
                }
            case sf::Event::MouseButtonPressed:
                // Only care for the left button
                if (event.mouseButton.button == sf::Mouse::Left) {
                    drawing = true;
                    brush.setPosition(event.mouseButton.x, event.mouseButton.y);
                    canvas.draw(brush);
                    canvas.display();
                }
                break;
            case sf::Event::MouseButtonReleased:
                if (event.mouseButton.button == sf::Mouse::Left)
                    drawing = false;
                break;
            case sf::Event::MouseMoved:
                if (drawing)
                {
                    brush.setPosition(event.mouseMove.x, event.mouseMove.y);
                    canvas.draw(brush);
                    canvas.display();
                }
                break;
            }
        }
        window.clear(Color(230,230,230));
        window.draw(sprite);
        window.display();
    }
}
void ErrorData(double err, double time) {
    ofstream fout("ErrorData.txt", ofstream::app);
    if (!fout.is_open()) {
        cout << "Error reading the ErrorData file" << endl;
        throw runtime_error("Invalid data!");
        system("pause");
    }
    fout << time << " " << err << endl;
    fout.close();
}
int main()
{
    NetWork hi;
    int size[n] = { input_n,128,10 };
    hi.SetLayers(n, size);
    data_info* data = new data_info[examples];

    for (int j = 0; j < examples; j++) data[j].pixels = new double[input_n];
    double ra = 0, right, predict, maxra = 0;
    int epoch = 0;

    bool study, repeat = true, universal_test=false;
    srand(time(NULL));
    cout << "Press G+V to save image. After close the window" << endl;
    while (repeat) {
        ShowWindow();
        cout << "STUDY? (1/0)" << endl;
        cin >> study;
        double start;
        double time1;
        int cc = 0;
        double tmp1;
        double ccc;
        start = clock();
        if (study) {
            read_data(data, examples, input_n);
            while (ra / examples * 100 < 100) {
                ra = 0;
                for (int i = 0; i < examples; i++) {
                    hi.SetInput(data[i].pixels);
                    right = data[i].digit;
                    predict = hi.forward_feed();
                    if (predict != right) {
                        hi.BackPropogation(right);
                        hi.WeightsUpdater(0.5);
                    }
                    else ra++;
                }
                time1 = clock();
                if (cc == 0) ccc = time1 - start;
                tmp1 = (time1 - start - ccc) / (1000 * 60);
                cc++;
                if (ra > maxra) maxra = ra;
                ErrorData(ra / examples * 100, tmp1);
                if (epoch == 20) break;
                cout << "ra: " << ra / examples * 100 << "\t" << "maxra: " << maxra / examples * 100 << "\t" << "epoch: " << epoch << endl;
                epoch++;
            }
            double end = clock();
            //double time = (end - start)/(CLOCKS_PER_SEC*60);
            cout << "TIME: " << time << endl;
            hi.SaveWeights();
        }
        else {
            hi.ReadWeights();
            cout << "Universal test? (1/0)" << endl;
            bool universal_test;
            cin >> universal_test;
            if (universal_test) {
                ifstream fin;
                fin.open("lib_10k.txt");
                int ex_tests = 10000;
                data_info* data_test = new data_info[ex_tests];
                for (int j = 0; j < ex_tests; j++) data_test[j].pixels = new double[input_n];

                for (int i = 0; i < ex_tests; i++) {
                    fin >> data_test[i].digit;
                    for (int j = 0; j < input_n; j++) fin >> data_test[i].pixels[j];
                }
                fin.close();
                double rA = 0;
                double right, predict;
                for (int i = 0; i < ex_tests; i++) {
                    hi.SetInput(data_test[i].pixels);
                    predict = hi.forward_feed();
                    right = data_test[i].digit;
                    if (right == predict) rA++;
                }
                cout << "RA: " << rA / ex_tests * 100 << endl;
            }
        }
        double* input = new double[input_n];
        read_test(input, input_n);
        hi.SetInput(input);
        bool flag = true;
        double tmp = hi.forward_feed(flag);
        cout << "Predict: " << tmp << endl;
        cout << "Repeat? (1/0)" << endl;
        cin >> repeat;
    }
    system("pause");
    return 0;
}
