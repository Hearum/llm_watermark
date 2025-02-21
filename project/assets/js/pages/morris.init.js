!(function (e) {
  "use strict";
  function a() {}
  (a.prototype.createLineChart = function (e, a, r, t, i, o) {
    Morris.Line({
      element: e,
      data: a,
      xkey: r,
      ykeys: t,
      labels: i,
      hideHover: "auto",
      gridLineColor: "rgba(108, 120, 151, 0.1)",
      resize: !0,
      lineColors: o,
      lineWidth: 2,
      parseTime: false // 添加此行以禁用时间解析
    });
  }),
    (a.prototype.createBarChart = function (e, a, r, t, i, o) {
      Morris.Bar({
        element: e,
        data: a,
        xkey: r,
        ykeys: t,
        labels: i,
        gridLineColor: "rgba(108, 120, 151, 0.1)",
        barSizeRatio: 0.4,
        resize: !0,
        hideHover: "auto",
        barColors: o,
      });
    }),
    (a.prototype.createAreaChart = function (e, a, r, t, i, o, b, y) {
      Morris.Area({
        element: e,
        pointSize: 0,
        lineWidth: 0,
        data: t,
        xkey: i,
        ykeys: o,
        labels: b,
        resize: !0,
        gridLineColor: "rgba(108, 120, 151, 0.1)",
        hideHover: "auto",
        lineColors: y,
        fillOpacity: 0.6,
        behaveLikeLine: !0,
      });
    }),
    (a.prototype.createDonutChart = function (e, a, r) {
      Morris.Donut({ element: e, data: a, resize: !0, colors: r });
    }),
    (a.prototype.createStackedChart = function (e, a, r, t, i, o) {
      Morris.Bar({
        element: e,
        data: a,
        xkey: r,
        ykeys: t,
        stacked: !0,
        labels: i,
        hideHover: "auto",
        barSizeRatio: 0.4,
        resize: !0,
        gridLineColor: "rgba(108, 120, 151, 0.1)",
        barColors: o,
      });
    }),
    (a.prototype.init = function () {
      this.createLineChart(
        "morris-line-example",
        [
          { y: "00:00", a: 120, b: 115, c: 100 },
          { y: "01:00", a: 135, b: 125, c: 90 },
          { y: "02:00", a: 147, b: 110, c: 115 },
          { y: "03:00", a: 155, b: 118, c: 103 },
          { y: "04:00", a:168, b: 112, c: 107 },
          { y: "05:00", a: 158, b: 122, c: 109 },
          { y: "06:00", a:148, b: 130, c: 90 },
          { y: "07:00", a: 120, b: 135, c: 92 },
          { y: "08:00", a: 120, b: 145, c: 125 },
          { y: "09:00", a: 112, b: 155, c: 130 },
          { y: "10:00", a: 102, b: 160, c: 95 },
          { y: "11:00", a: 90, b: 165, c: 80 },
          { y: "12:00", a: 87, b: 170, c: 45 },
          { y: "13:00", a: 65, b: 175, c: 50 },
          { y: "14:00", a: 88, b: 180, c: 55 },
          { y: "15:00", a: 120, b: 185, c: 60 },
          { y: "16:00", a: 124, b: 180, c: 55 },
          { y: "17:00", a: 124, b: 175, c: 52 },
          { y: "18:00", a: 121, b: 170, c: 45 },
          { y: "19:00", a: 132, b: 165, c: 41 },
          { y: "20:00", a: 121, b: 160, c: 35 },
          { y: "21:00", a: 132, b: 155, c: 38 },
          { y: "22:00", a: 126, b: 150, c: 28 },
          { y: "23:00", a: 110, b: 145, c: 20 }
        ],
        "y",
        ["a", "b", "c"],
        ["九月拦截流量", "十月拦截流量", "十一月拦截流量"],
        ["#ccc", "#7a6fbe", "#28bbe3"]
      );
      this.createLineChart(
        "morris-line-example-2",
        [
          { y: "5月", a: 50, b: 80, c: 20 },
          { y: "6月", a: 130, b: 100, c: 80 },
          { y: "7月", a: 80, b: 60, c: 70 },
          { y: "8月", a: 70, b: 200, c: 140 },
          { y: "9月", a: 180, b: 140, c: 150 },
          { y: "10月", a: 105, b: 100, c: 80 },
          { y: "11月", a: 250, b: 150, c: 200 },
        ],
        "y",
        ["a", "b", "c"],
        ["Activated", "Pending", "Deactivated"],
        ["#ccc", "#7a6fbe", "#28bbe3"]
      );
      this.createBarChart(
        "morris-bar-example",
        [
          { y: "2009", a: 100, b: 90 },
          { y: "2010", a: 75, b: 65 },
          { y: "2011", a: 50, b: 40 },
          { y: "2012", a: 75, b: 65 },
          { y: "2013", a: 50, b: 40 },
          { y: "2014", a: 75, b: 65 },
          { y: "2015", a: 100, b: 90 },
          { y: "2016", a: 90, b: 75 },
        ],
        "y",
        ["a", "b"],
        ["Series A", "Series B"],
        ["#7a6fbe", "#28bbe3"]
      );
      this.createAreaChart(
        "morris-area-example",
        0,
        0,
        [
          { y: "2007", a: 0, b: 0, c: 0 },
          { y: "2008", a: 150, b: 45, c: 15 },
          { y: "2009", a: 60, b: 150, c: 195 },
          { y: "2010", a: 180, b: 36, c: 21 },
          { y: "2011", a: 90, b: 60, c: 360 },
          { y: "2012", a: 75, b: 240, c: 120 },
          { y: "2013", a: 30, b: 30, c: 30 },
        ],
        "y",
        ["a", "b", "c"],
        ["Series A", "Series B", "Series C"],
        ["#ccc", "#7a6fbe", "#28bbe3"]
      );
      this.createDonutChart(
        "morris-donut-example",
        [
          { label: "macOS", value: 1600 },
          { label: "Windows", value: 1400 },
          { label: "Linux", value: 65 },
        ],
        ["#f0f1f4", "#7a6fbe", "#28bbe3"]
      );
      this.createStackedChart(
        "morris-bar-stacked",
        [
          { y: "2005", a: 45, b: 180 },
          { y: "2006", a: 75, b: 65 },
          { y: "2007", a: 100, b: 90 },
          { y: "2008", a: 75, b: 65 },
          { y: "2009", a: 100, b: 90 },
          { y: "2010", a: 75, b: 65 },
          { y: "2011", a: 50, b: 40 },
          { y: "2012", a: 75, b: 65 },
          { y: "2013", a: 50, b: 40 },
          { y: "2014", a: 75, b: 65 },
          { y: "2015", a: 100, b: 90 },
          { y: "2016", a: 80, b: 65 },
        ],
        "y",
        ["a", "b"],
        ["Series A", "Series B"],
        ["#7a6fbe", "#f0f1f4"]
      );
    }),
    (e.MorrisCharts = new a()),
    (e.MorrisCharts.Constructor = a);
})(window.jQuery),
  (function () {
    "use strict";
    window.jQuery.MorrisCharts.init();
  })();
