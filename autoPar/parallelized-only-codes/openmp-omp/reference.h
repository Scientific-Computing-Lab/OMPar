bool correctResult(int *data, const int n, const int b, const int repeat) {
  for (int i = 0; i < n; i++) {
    int sum = 0;
    for (int j = 0; j < repeat; j++)
      sum += j % b;
    if (sum + i != data[i]) {
      printf("check: %d != %d\n", data[i], sum);
      return false;
    }
  }
  return true;
}
