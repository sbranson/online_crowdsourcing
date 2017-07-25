from scipy.spatial.distance import pdist, squareform
from numpy.random import np
from numpy import concatenate
import heapq

# Solves a facility location problem using the algorithm from Jain et al. "Greedy Facility Location Algorithms Analyzed
# using Dual Fitting with Factor-Revealing LP".  A facility location algorithm takes as input a graph G=(V,E), where
# each edge is a cost for connecting two vertices (or "cities").  The goal is to choose a subset of vertices (or 
# "facilities"), such that each city is connected to exactly one facility, while minimizing the total cost.  The
# cost is the sum of selected edge connections, plus predefined cost for opening up each facility.  The algorithm
# of Jain et al. is an O(E log E) algorithm with an approximation guarantee of within 1.61 times the optimal 
# solution
#
# Let cityFacilityCosts be a list of 3-tuples, where each element (cost,facility,city) is the cost
# of connecting a city to a facility and openFacilityCost is the cost of opening a new facility.
# Returns a dict where retval[facility][city] is present for connected facility-city pairs, with value equal 
# to the connection cost.
class FacilityLocation():
    def __init__(self, points=None, cityFacilityCosts=None, cityDisallowedCityNeighbors=None):
        self.points = points
        self.cityFacilityCosts = cityFacilityCosts
        self.costs_s = None
        self.allFacilities = {}
        self.allCities = {}
        self.city_disallowed_city_neighbors = cityDisallowedCityNeighbors
        if not self.points is None:
            for i in range(0,len(self.points)):
                self.allFacilities[i] = self.allCities[i] = 1
        else:
            for c in cityFacilityCosts:
                if not c[1] in self.allFacilities: self.allFacilities[c[1]] = 1
                if not c[2] in self.allCities: self.allCities[c[2]] = 1

    def solve(self, openFacilityCost=None, openFacilityCosts=None, debug=0):
        if openFacilityCosts is None:
            openFacilityCosts = {}
            for f in self.allFacilities:
                openFacilityCosts[f] = openFacilityCost
        self.openFacilityCosts = openFacilityCosts
        if self.cityFacilityCosts is None:
            self.compute_costs()

        # A priority queue keyed by facility, with value set to the value of alpha when that facility should be opened
        self.fac_open_candidates = priority_dict() 

        # A dict keyed by facility, where each value is a 4-tuple: (S1,S2,n1,n2), where S1 is the sum of offers from 
        # unconnected cities--excluding alpha terms--to this facility, n1 is the number of offers from unconnected 
        # cities, S2 is the sum of switching offers from connected cities, and n2 is the number of offers from
        # connected cities.  Keeping this structure allows us to compute at any given time the value of alpha
        # at which a facility should be opened
        self.facility_offers = {} 

        self.facility_cities = {} # Return value, a dict where the key is a facility and each value is a list of connected cities
        self.city_facilities = {} # A dict keyed by city, with the value set to the facility that city is connected to
        self.fac_city_offers = {} # For each facility f and city c, fac_city_offers[f][c] is an offer to connect c to f
        self.city_fac_offers = {} # For each facility f and city c, city_fac_offers[c][f] is an offer to connect c to f
        self.facility_disallowed_cities = {} # For each facility f and city c, facility_disallowed_cities[f][c] implies c cannot be connected to f
        self.total_cost = 0

        # Now sort the object-object pairs by increasing matching cost, and iterate through each one.  This enables us
        # to monotonically increase alpha, while keeping track of exactly the connection offers that are non-zero
        if self.costs_s is None:
            self.costs_s = sorted(self.cityFacilityCosts, key=lambda t: t[0]) 
        self.big = 0
	for k in range(0, len(self.costs_s)):
		if len(self.costs_s[k]) > 0:
			self.big = max(self.big, self.costs_s[k][0] + 1)
        for k in range(0, len(self.costs_s)):
            cost = self.costs_s[k][0]
            
            # For some unopened facility, the total offers it receives reaches the cost of opening a facility before
            # alpha=cost.  Open that facility
            while len(self.fac_open_candidates)>0 and (not self.costs_s[k][2] in self.city_facilities):
                alpha = self.fac_open_candidates[self.fac_open_candidates.smallest()]
                if alpha > cost:
                    break
                self.open_facility(self.fac_open_candidates.pop_smallest(), alpha, debug)
            
            if k < len(self.costs_s):
                if debug > 1: self.debug_offers()
                facility = self.costs_s[k][1]
                city = self.costs_s[k][2]
                self.offer(city, facility, cost, debug)
                if debug > 1: self.debug_offers()
        while len(self.city_facilities) < len(self.allCities):
            if debug > 0: print "Opening facility because some cities were unconnected"
            alpha = self.fac_open_candidates[self.fac_open_candidates.smallest()]
            self.open_facility(self.fac_open_candidates.pop_smallest(), alpha, debug)
        return (self.facility_cities, self.total_cost)
            
    def offer(self, city, facility, cost, debug):
        #if 3 in self.facility_disallowed_cities: print "      facility_disallowed_cities[3]="+str(self.facility_disallowed_cities[3])
        if (not city in self.city_facilities):
            if (not facility in self.facility_disallowed_cities or
                not city in self.facility_disallowed_cities[facility] or 
                self.facility_disallowed_cities[facility][city][0] <= 0):  
                    # Offer city as a candidate connection to facility
                    if debug > 0: print "Offer city " + str(city) + " to facility " + str(facility) + " at cost " + str(cost)
                    if not facility in self.facility_offers:
                        self.facility_offers[facility] = [0,0,0,0]
                    t = self.facility_offers[facility]
                    t[0] -= cost
                    t[2] += 1
                    if not city in self.city_fac_offers: self.city_fac_offers[city]={}
                    if not facility in self.fac_city_offers: self.fac_city_offers[facility]={}
                    self.fac_city_offers[facility][city] = cost
                    self.city_fac_offers[city][facility] = (cost,None,None)
                    if not facility in self.facility_cities:
                        if t[2] != 0:
                            self.fac_open_candidates[facility] = (self.openFacilityCosts[facility]-t[0]-t[1])/t[2]
                        elif facility in self.fac_open_candidates:
                            del self.fac_open_candidates[facility]
                    if facility in self.facility_cities:
                        # Connect city to existing facility
                        if debug > 0: print "Connect city " + str(city) + " to existing facility " + str(facility) + " at cost " + str(cost)
                        self.connect(facility, city, debug)    
                    else:
                        if (not self.city_disallowed_city_neighbors is None) and (city in self.city_disallowed_city_neighbors):
                            for c2 in self.city_disallowed_city_neighbors[city]:
                                if not isinstance(self.city_disallowed_city_neighbors[city],dict) or facility!=self.city_disallowed_city_neighbors[city][c2]:
                                    # Connecting city to facility requires that c2 can't be connected to facility
                                    if not facility in self.facility_disallowed_cities: self.facility_disallowed_cities[facility] = {}
                                    if not c2 in self.facility_disallowed_cities[facility]:
                                        self.facility_disallowed_cities[facility][c2] = [0,0]
                                    self.facility_disallowed_cities[facility][c2][0] += 1   
                                    if debug > 0: print "  offer " + str(city) + " disallow " + str(c2) + " from " + str(facility) + " count=" + str(self.facility_disallowed_cities[facility][c2][0])
            elif (facility in self.facility_disallowed_cities and city in self.facility_disallowed_cities[facility] and
                  self.facility_disallowed_cities[facility][city][0] > 0):
                if debug > 0:  print "disallowed offer city " + str(city) + " to facility " + str(facility) + " at cost " + str(cost)
                self.facility_disallowed_cities[facility][city][1] = cost
    
    def open_facility(self, facility, alpha, debug):
        # Open a new facility and connect all candidate cities
        self.facility_cities[facility] = {}
        if not facility in self.facility_disallowed_cities: self.facility_disallowed_cities[facility] = {}
        self.total_cost += self.openFacilityCosts[facility]
        if debug > 0: print "Open facility " + str(facility) + " at alpha " + str(alpha) + ":"
        offers = [c for c in self.fac_city_offers[facility]]
        for c in self.fac_city_offers[facility]:
            # Connect each offering city c to facility
            self.connect(facility, c, debug)
        del self.fac_city_offers[facility] 
    
    def connect(self, facility, c, debug):
        #if 3 in self.facility_disallowed_cities: print "      facility_disallowed_cities[3]="+str(self.facility_disallowed_cities[3])
        # Connect facility to city c.  Assumes a connection between them has already been offered
        c_ip_j = self.fac_city_offers[facility][c] 
        if debug > 0: print "  connect city " + str(c) + " to facility " + str(facility) + " at cost " + str(c_ip_j)
        if c in self.facility_disallowed_cities[facility] and self.facility_disallowed_cities[facility][c][0] > 0: raise Exception("huh?")
        del self.city_fac_offers[c][facility]
        self.facility_cities[facility][c] = c_ip_j
        self.total_cost += c_ip_j

        '''
        if (not self.city_disallowed_city_neighbors is None) and (c in self.city_disallowed_city_neighbors):
            for c2 in self.city_disallowed_city_neighbors[c]:
              if not isinstance(self.city_disallowed_city_neighbors[c],dict) or facility!=self.city_disallowed_city_neighbors[c][c2]:
                # Connecting city to facility requires that c2 can't be connected to facility
                if not c2 in self.facility_disallowed_cities[facility]:
                    self.facility_disallowed_cities[facility][c2] = [0,0]
                self.facility_disallowed_cities[facility][c2][0] += 1   
                if debug > 0: print "  connect " + str(c) + " disallow " + str(c2) + " from " + str(facility) + " count=" + str(self.facility_disallowed_cities[facility][c2][0])
        '''
        
        if c in self.city_facilities:
            # If c is already connected to a facility f, we must switch it
            f=self.city_facilities[c]
            self.total_cost -= self.facility_cities[f][c]
            if debug > 0: print "    requires switch from facility " + str(f) + " at cost " + str(self.facility_cities[f][c])
            del self.facility_cities[f][c]
            del self.city_facilities[c]
        
        self.city_facilities[c] = facility
        bad = []
        for f in self.city_fac_offers[c]:
            # Update the offers of city c to switch to other unopened facilities
            c_i_j = self.fac_city_offers[f][c]
            t = self.facility_offers[f]
            if not self.city_fac_offers[c][f][1] is None:
                t[1] += self.city_fac_offers[c][f][2]  # remove c from sum offers to connect unconnected cities to f
                t[3] -= 1             # number of offers to connect unconnected cities to f
                if t[3] < 0 and debug > 1: raise Exception("huh?")
            else:
                t[0] += c_i_j         # remove c from sum switch offers 
                t[2] -= 1             # number of switch
                if t[2] < 0 and debug > 1: raise Exception("huh?")
            if (not f in self.facility_cities) and (c_i_j < c_ip_j):
                if (not f in self.facility_disallowed_cities or
                    not c in self.facility_disallowed_cities[f] or 
                    self.facility_disallowed_cities[f][c][0] <= 0):  
                    if debug > 0: print "    offer switch from facility " + str(facility) + "("+str(c_ip_j)+ ") to " + str(f) + "(" + str(c_i_j)+") for city " + str(c)
                    t[1] += c_ip_j-c_i_j  # sum offers to switch connections of already connected cities to f
                    t[3] += 1             # number of offers to switch connections of already connected cities to f
                    self.city_fac_offers[c][f] = (c_i_j,f,c_i_j-c_ip_j)
                    if (not self.city_disallowed_city_neighbors is None) and (c in self.city_disallowed_city_neighbors):
                            for c2 in self.city_disallowed_city_neighbors[c]:
                                if not isinstance(self.city_disallowed_city_neighbors[c],dict) or facility!=self.city_disallowed_city_neighbors[c][c2]:
                                    # Connecting city to facility requires that c2 can't be connected to facility
                                    if not f in self.facility_disallowed_cities: self.facility_disallowed_cities[f] = {}
                                    if not c2 in self.facility_disallowed_cities[f]:
                                        self.facility_disallowed_cities[f][c2] = [0,0]
                                    self.facility_disallowed_cities[f][c2][0] += 1   
                                    if debug > 0: print "  offer switch " + str(c) + " disallow " + str(c2) + " from " + str(f) + " count=" + str(self.facility_disallowed_cities[f][c2][0])
            else:
                bad.append(f)
                if (not self.city_disallowed_city_neighbors is None) and (c in self.city_disallowed_city_neighbors) and f != facility:
                    for c2 in self.city_disallowed_city_neighbors[c]:
                        if c2 in self.facility_disallowed_cities[f]:
                            self.facility_disallowed_cities[f][c2][0] -= 1
                            if self.facility_disallowed_cities[f][c2][0] == 0 and self.facility_disallowed_cities[f][c2][1] != 0 and not c2 in self.city_facilities:
                                if debug > 0: print "  undisallow " + str(c2) + " from " + str(f) + " ( "+str(c)+" was connected) cost=" + str(self.facility_disallowed_cities[f][c2][1])
                                self.offer(c2, f, self.facility_disallowed_cities[f][c2][1], debug)

            if (not f in self.facility_cities):
                self.fac_open_candidates[f] = (self.openFacilityCosts[f]-t[0]-t[1])/t[2] if t[2] > 0 else self.big
            
        for f in bad:
            del self.fac_city_offers[f][c]
            del self.city_fac_offers[c][f]
    
    def compute_costs(self, type='euclidean'):
        C = squareform(pdist(self.points, type))
        self.cityFacilityCosts = []
        for i in range(0,C.shape[0]):
            for j in range(0,C.shape[0]):
                self.cityFacilityCosts.append((C[i,j], i,j))
        self.costs_s = None
        
    def debug_offers(self):
        print ""
        print "--facility_cities " + str(self.facility_cities)
        print "--city_facilities " + str(self.city_facilities)
        print "--fac_open_candidates " + str(self.fac_open_candidates)
        print "--fac_city_offers " + str(self.fac_city_offers)
        print "--city_fac_offers " + str(self.city_fac_offers)
        print "--facility_offers " + str(self.facility_offers)

class priority_dict(dict):
    """Dictionary that can be used as a priority queue.

    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is
    that priorities of items can be efficiently updated (amortized O(1))
    using code as 'thedict[item] = new_priority.'

    The 'smallest' method can be used to return the object with lowest
    priority, and 'pop_smallest' also removes it.

    The 'sorted_iter' method provides a destructive sorted iterator.
    """
    
    def __init__(self, *args, **kwargs):
        super(priority_dict, self).__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in self.iteritems()]
        heapq.heapify(self._heap)

    def smallest(self):
        """Return the item with the lowest priority.

        Raises IndexError if the object is empty.
        """
        
        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heapq.heappop(heap)
            v, k = heap[0]
        return k

    def pop_smallest(self):
        """Return the item with the lowest priority and remove it.

        Raises IndexError if the object is empty.
        """
        
        heap = self._heap
        v, k = heapq.heappop(heap)
        while k not in self or self[k] != v:
            v, k = heapq.heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        # We are not going to remove the previous value from the heap,
        # since this would have a cost O(n).
        
        super(priority_dict, self).__setitem__(key, val)
        
        if len(self._heap) < 2 * len(self):
            heapq.heappush(self._heap, (val, key))
        else:
            # When the heap grows larger than 2 * len(self), we rebuild it
            # from scratch to avoid wasting too much memory.
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        # Reimplementing dict.update is tricky -- see e.g.
        # http://mail.python.org/pipermail/python-ideas/2007-May/000744.html
        # We just rebuild the heap from scratch after passing to super.
        
        super(priority_dict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def sorted_iter(self):
        """Sorted iterator of the priority dictionary items.

        Beware: this will destroy elements as they are returned.
        """
        
        while self:
            yield self.pop_smallest()



def filter_by_perm(user, queryset, perm):
    bad_ids = []
    for a in queryset:
        if not user.has_perm('read', a):
            bad_ids.append(a.pk)
    return queryset.exclude(id__in=bad_ids) if len(bad_ids) > 0 else queryset



'''
import matplotlib.pyplot as plt
import math
def test_facility_location(numPts=20, ptDim=2, openCosts=None, p1=None, p2=None):
    if openCosts is None:
        openCosts = [1,2,4,8,16,32,64,128,256]
        for i in range(0,len(openCosts)): openCosts[i] *= numPts/20.0
    w = math.ceil(math.sqrt(len(openCosts)))
    fig = plt.figure()
    if p1 is None: p1 = 2.5 * np.random.randn(numPts, ptDim) + 3
    if p2 is None: p2 = 2.5 * np.random.randn(numPts, ptDim) + -3
    p = np.concatenate((p1,p2))
    fac = FacilityLocation(points=p)
    for n in range(0,len(openCosts)):
        [facilities,cost]=fac.solve(openFacilityCost=openCosts[n])
        ax = fig.add_subplot(100*w + 10*w + n+1)
        pt1 = ax.plot(p1[:,0], p1[:,1], 'b.')
        pt2 = ax.plot(p2[:,0], p2[:,1], 'r.')
        for i in facilities:
            for j in facilities[i]:
                ax.plot([p[i,0],p[j,0]], [p[i,1],p[j,1]], 'g')
        ax.set_title('openCost='+str(openCosts[n]) + " totalCost="+str(cost))
    fig.show()
    return (p1,p2)
'''
